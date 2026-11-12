"""Words built separately must work together."""

import pytest

import plotsimple as ps


def test_second_y_with_legend_bottom_keeps_both_series(marks):
    chart = ps.chart(marks)
    chart.lines(x="test", y="marks", colour="class")
    chart.second_y("height", drawn_as="lines", label="Height (cm)")
    chart.legend_bottom()
    fig = chart._render()
    labels = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert "A" in labels and "B" in labels
    assert any("height" in l.lower() for l in labels)
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_full_sentence_renders(marks, tmp_path):
    chart = ps.chart(marks)
    chart.bars(x="name", y="marks", colour="class", error="sd")
    chart.title_top("My class marks")
    chart.label_x("Student")
    chart.label_y("Marks")
    chart.line_at_y(50)
    chart.shade_between_y(40, 60)
    chart.range_y(0, 110)
    chart.numbers_on_bars()
    chart.caption_bottom("Made-up data")
    chart.legend_right()
    chart.style("school")
    chart.save(tmp_path / "all.png")
    assert (tmp_path / "all.png").stat().st_size > 0


def test_page_of_different_charts(marks, tmp_path):
    a = ps.chart(marks).bars(x="name", y="marks").title_top("Bars")
    b = ps.chart(marks).dots(x="height", y="marks").fit_line(confidence_band=True).title_top("Fit")
    c = ps.chart(marks).histogram(x="marks").title_top("Spread")
    d = ps.chart(marks).heatmap(x="name", y="test", value="marks", numbers=True)
    ps.grid(a, b, c, d, per_row=2).title_top("Everything").save(tmp_path / "page.png")
    assert (tmp_path / "page.png").stat().st_size > 0


def test_grid_with_letters_labels_each_chart(tmp_path):
    marks, weather = ps.example("marks"), ps.example("weather")
    a = ps.chart(marks).bars(x="class", y="marks")
    b = ps.chart(weather).lines(x="month", y="rain")
    c = ps.chart(marks).histogram(x="marks")
    page = ps.grid(a, b, c, per_row=2, letters=True)
    fig = page._render()
    assert [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")] == ["A", "B", "C"]
    page.save(tmp_path / "grid.png")
    assert (tmp_path / "grid.png").stat().st_size > 0
    fig = ps.side_by_side(a, b)._render()
    assert not [ax for ax in fig.axes if ax.get_title(loc="left")]  # no letters unless asked


def test_one_above_other_with_letters(tmp_path):
    marks = ps.example("marks")
    a = ps.chart(marks).boxes(x="class", y="marks")
    b = ps.chart(marks).dots(x="hours", y="marks")
    fig = ps.one_above_other(a, b, letters=True)._render()
    assert [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")] == ["A", "B"]


def test_page_letters_must_be_true_or_false():
    a = ps.chart(ps.example("marks")).bars(x="class", y="marks")
    with pytest.raises(ps.PlotsimpleError, match="letters=True or letters=False"):
        ps.side_by_side(a, a, letters="yes")
    with pytest.raises(ps.PlotsimpleError, match="letters=True or letters=False"):
        ps.grid(a, a, letters=1)
