import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata._stats import t_interval


def test_bars_draw(marks):
    chart = sdata.chart(marks)
    chart.bars(x="name", y="marks")
    fig = chart._render()
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 4


def test_bars_show_average_matches_pandas(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks")
    heights = [p.get_height() for p in chart._render().axes[0].patches]
    expected = marks.groupby("name", sort=False)["marks"].mean().tolist()
    assert np.allclose(heights, expected)


def test_order_of_steps_does_not_matter(marks):
    a = sdata.chart(marks)
    a.title_top("Hello")
    a.bars(x="name", y="marks")
    b = sdata.chart(marks)
    b.bars(x="name", y="marks")
    b.title_top("Hello")
    assert a._render().axes[0].get_title() == b._render().axes[0].get_title() == "Hello"


def test_lists_work_without_table():
    chart = sdata.chart()
    chart.bars(x=["a", "b", "c"], y=[3, 5, 2])
    assert len(chart._render().axes[0].patches) == 3


def test_single_list_is_y():
    chart = sdata.chart().lines([3, 5, 2])
    line = chart._render().axes[0].lines[0]
    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [3, 5, 2]


def test_misspelled_column_suggests(marks):
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        sdata.chart(marks).bars(x="name", y="mraks")


def test_missing_word_suggests(marks):
    with pytest.raises(SimplydataError, match=r"\.title_top\(\)"):
        sdata.chart(marks).title_topp("x")


def test_missing_word_is_attribute_error(marks):
    assert not hasattr(sdata.chart(marks), "no_such_step")


def test_unknown_option_suggests(marks):
    with pytest.raises(SimplydataError, match="colour="):
        sdata.chart(marks).bars(x="name", y="marks", colur="class")


def test_color_spelling_accepted(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks", color="class")
    assert chart._render().axes[0].get_legend() is not None


def test_colour_name_is_single_colour(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks", colour="red")
    face = chart._render().axes[0].patches[0].get_facecolor()
    assert np.allclose(face[:3], (1, 0, 0))


def test_wrong_choice_suggests(marks):
    with pytest.raises(SimplydataError, match='show="average"'):
        sdata.chart(marks).bars(x="name", y="marks", show="averag")


def test_error_ci_is_t_interval():
    v = [10.0, 12.0, 11.0, 15.0]
    lo, hi = t_interval(v)
    ref = stats.t.interval(0.95, len(v) - 1, loc=np.mean(v), scale=stats.sem(v))
    assert np.allclose((lo, hi), ref)


def test_error_sd_uses_n_minus_1(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks", error="sd")
    ax = chart._render().axes[0]
    first = marks[marks["name"] == "Asha"]["marks"]
    bar_line = ax.lines[0]
    ys = bar_line.get_ydata()
    lo, hi = min(ys), max(ys)
    assert np.isclose(hi - lo, 2 * first.std(ddof=1))


def test_stacked_needs_colour(marks):
    with pytest.raises(SimplydataError, match="stacked"):
        sdata.chart(marks).bars(x="name", y="marks", stacked=True)


def test_stacked_and_sideways(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks", colour="class", stacked=True, sideways=True)
    assert len(chart._render().axes[0].patches) == 8


def test_save_png_and_pdf(marks, tmp_path):
    chart = sdata.chart(marks).dots(x="height", y="marks")
    chart.save(tmp_path / "a.png")
    chart.save(tmp_path / "a.pdf")
    assert (tmp_path / "a.png").stat().st_size > 0
    assert (tmp_path / "a.pdf").stat().st_size > 0


def test_save_needs_file_type(marks, tmp_path):
    with pytest.raises(SimplydataError, match=".png"):
        sdata.chart(marks).dots(x="height", y="marks").save(tmp_path / "a")


def test_repr_png(marks):
    png = sdata.chart(marks).bars(x="name", y="marks")._repr_png_()
    assert png[:4] == b"\x89PNG"


def test_read_excel_and_wrong_sheet(marks, tmp_path):
    path = tmp_path / "marks.xlsx"
    marks.to_excel(path, sheet_name="2026", index=False)
    assert len(sdata.chart(path, sheet="2026")._data) == len(marks)
    with pytest.raises(SimplydataError, match='Did you mean "2026"'):
        sdata.chart(path, sheet="2025")


def test_missing_file_suggests(marks, tmp_path):
    marks.to_csv(tmp_path / "marks.csv", index=False)
    with pytest.raises(SimplydataError, match="marks.csv"):
        sdata.chart(tmp_path / "mark.csv")


def test_polars_input(marks):
    pl = pytest.importorskip("polars")
    chart = sdata.chart(pl.from_pandas(marks)).bars(x="name", y="marks")
    assert len(chart._render().axes[0].patches) == 4


def test_fast_uses_polars(marks, tmp_path):
    pytest.importorskip("polars")
    marks.to_csv(tmp_path / "m.csv", index=False)
    data = sdata.chart(tmp_path / "m.csv", fast=True)._data
    pd.testing.assert_frame_equal(data.reset_index(drop=True), marks, check_dtype=False)


def test_style_and_size(marks):
    chart = sdata.chart(marks).bars(x="name", y="marks").style("school").size(width=10, height=4)
    fig = chart._render()
    assert tuple(fig.get_size_inches()) == (10, 4)
    with pytest.raises(SimplydataError, match='"school"'):
        chart.style("scool")


def test_layers_stack(marks):
    chart = sdata.chart(marks)
    chart.bars(x="test", y="marks")
    chart.lines(x="test", y="height")
    ax = chart._render().axes[0]
    assert len(ax.patches) == 3 and len(ax.lines) >= 1


def test_assigning_tells_how(marks):
    chart = sdata.chart(marks)
    with pytest.raises(SimplydataError, match=r"chart\.title_top\('Hi'\)"):
        chart.title = "Hi"


def test_dates_get_short_labels():
    from matplotlib.dates import ConciseDateFormatter
    days = pd.date_range("2026-01-01", periods=120, freq="D")
    chart = sdata.chart(pd.DataFrame({"day": days, "temp": np.arange(120.0)})).lines(x="day", y="temp")
    ax = chart._render().axes[0]
    assert isinstance(ax.xaxis.get_major_formatter(), ConciseDateFormatter)
