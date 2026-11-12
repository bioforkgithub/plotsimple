"""Old names keep working for one release, with one gentle note the first time they are used.

The conftest resets the notes before each test, so each test starts like a new session.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import plotsimple as ps
from plotsimple import PlotsimpleError

NOTE_END = "The old name will stop working in a later version."


def note(old, new):
    return f"Note: {old} is now called {new}. {NOTE_END}"


@pytest.fixture
def gappy():
    return pd.DataFrame({"name": ["a", "b", "c"], "marks": [50.0, None, 70.0]})


def notes(out):
    return [line for line in out.splitlines() if line.endswith(NOTE_END)]


# ---------------------------------------------------------------- table words
@pytest.mark.parametrize("old, new, call", [
    ("show_first", "show_first_rows", lambda t: t.show_first(2)),
    ("show_last", "show_last_rows", lambda t: t.show_last(2)),
    ("count_empty", "count_empty_cells", lambda t: t.count_empty()),
    ("remove_empty_rows", "remove_rows_with_empty_cells", lambda t: t.remove_empty_rows("marks")),
    ("fill_empty", "fill_empty_cells", lambda t: t.fill_empty("marks", with_value=0)),
])
def test_old_table_words_work_and_note_once(old, new, call, gappy, capsys):
    a, b = ps.table(gappy), ps.table(gappy)
    call(a)
    call(b)
    out = capsys.readouterr().out
    assert notes(out) == [note(f'"{old}"', f'"{new}"')]  # once, not twice
    getattr(ps.table(gappy), new)  # the new word exists
    if old in ("remove_empty_rows", "fill_empty"):
        assert len(a._history) == 1 and a._frame.to_pandas().equals(b._frame.to_pandas())


def test_old_done_still_gives_the_stored_table(gappy, capsys):
    plain = ps.table(gappy).done()
    assert isinstance(plain, pd.DataFrame)
    assert notes(capsys.readouterr().out) == [note('"done"', '"as_pandas"')]
    pl = pytest.importorskip("polars")
    assert isinstance(ps.table(pl.from_pandas(gappy)).done(), pl.DataFrame)  # as before


def test_as_pandas_and_as_polars(gappy):
    t = ps.table(gappy)
    assert isinstance(t.as_pandas(), pd.DataFrame)
    pl = pytest.importorskip("polars")
    assert isinstance(t.as_polars(), pl.DataFrame)
    fast = ps.table(pl.from_pandas(gappy))
    assert isinstance(fast.as_pandas(), pd.DataFrame) and isinstance(fast.as_polars(), pl.DataFrame)


def test_as_polars_without_polars_is_explained(gappy, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_polars(name, *args, **kwargs):
        if name == "polars":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_polars)
    with pytest.raises(PlotsimpleError, match=r"pip install plotsimple\[fast\]"):
        ps.table(gappy).as_polars()


def test_unknown_word_is_still_an_error(gappy):
    with pytest.raises(AttributeError, match="no step called"):
        ps.table(gappy).show_firsts()
    assert not hasattr(ps.table(gappy), "show_firsts")


# ---------------------------------------------------------------- chart words and options
def test_old_chart_words_work(marks, capsys):
    chart = ps.chart(marks).boxes(x="class", y="marks").all_points(x="class", y="marks")
    assert [s.name for s in chart._steps][-1] == "all_dots"
    c = ps.chart(marks).heatmap(x="test", y="name", value="marks").colour_bar_label("Marks")
    assert c._render()  # draws
    ps.chart(marks).heatmap(x="test", y="name", value="marks").color_bar_label("Marks")
    one = ps.chart(marks).dots(x="height", y="marks").one_per(row="class", column="test")
    assert len(one._render().axes) == 2 * 3
    out = capsys.readouterr().out
    assert notes(out) == [note('"all_points"', '"all_dots"'), note('"colour_bar_label"', '"label_colour_bar"'),
                          note('"color_bar_label"', '"label_colour_bar"'), note('"one_per"', '"one_chart_per"'),
                          note('"row="', '"down="'), note('"column="', '"across="')]
    plt.close("all")


def test_old_options_work_with_a_note(marks, tmp_path, capsys):
    ps.chart(marks).dots(x="height", y="marks").fit_line(band=True)._render()
    weather = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "temp": [2, 4, 8], "rain": [50, 40, 30]})
    second = ps.chart(weather).lines(x="month", y="temp").second_y("rain", chart="bars")
    assert second._steps[-1].kwargs["drawn_as"] == "bars"
    a = ps.chart(marks).bars(x="name", y="marks")
    assert ps.grid(a, a, a, columns=3)._columns == 3
    assert ps.chart(marks).bars(x="name", y="marks").one_chart_per("class", columns=1)._settings["split"]["columns"] == 1
    marks.to_csv(tmp_path / "m.csv", index=False)
    pytest.importorskip("polars")
    assert ps.table(tmp_path / "m.csv", big_data=True)._engine == "Polars"
    ps.chart(tmp_path / "m.csv", big_data=True)
    out = notes(capsys.readouterr().out)
    assert out == [note('"band="', '"confidence_band="'), note('"chart="', '"drawn_as="'),
                   note('"columns="', '"per_row="'), note('"big_data="', '"fast="')]
    plt.close("all")


def test_old_and_new_option_together_is_explained(marks):
    with pytest.raises(PlotsimpleError, match="only once"):
        ps.chart(marks).dots(x="height", y="marks").fit_line(band=True, confidence_band=True)


def test_old_zoom_box_options(marks, capsys):
    chart = ps.chart(marks).dots(x="height", y="marks").zoom_box(x_from=140, x_to=160, y_from=40, y_to=80,
                                                                where="upper left")
    step = chart._steps[-1]
    assert step.kwargs["x_between"] == (140, 160) and step.kwargs["y_between"] == (40, 80)
    assert step.kwargs["where"] == "top left"
    assert notes(capsys.readouterr().out) == [
        note('"x_from= and x_to="', '"x_between="'), note('"y_from= and y_to="', '"y_between="'),
        note('where="upper left"', 'where="top left"')]
    chart._render()
    plt.close("all")


# ---------------------------------------------------------------- new words
def test_show_legend_brings_the_legend_back(marks):
    chart = ps.chart(marks).lines(x="test", y="marks", colour="class").hide_legend()
    assert not chart._render().legends and chart._render().axes[0].get_legend() is None
    chart.show_legend()
    fig = chart._render()
    assert fig.legends or fig.axes[0].get_legend() is not None
    plt.close("all")


@pytest.mark.parametrize("long, short", [("standard deviation", "sd"), ("standard error", "se"),
                                         ("confidence interval", "ci")])
def test_long_error_names_are_the_same_as_short(marks, long, short, capsys):
    a = ps.chart(marks).bars(x="class", y="marks", error=long)
    b = ps.chart(marks).bars(x="class", y="marks", error=short)
    assert a._steps[-1].kwargs["error"] == b._steps[-1].kwargs["error"] == short
    out = capsys.readouterr().out.splitlines()
    assert out[0] == out[1] and out[0].startswith("Error bars: the ")  # says which one was drawn
    assert notes("\n".join(out)) == []  # short forms are standard abbreviations: no note
    e = ps.chart(marks).dots(x="test", y="marks").error_bars(long.title())  # any capitals work
    assert e._steps[-1].kwargs["kind"] == short
    plt.close("all")


def test_error_choice_lists_the_long_names(marks):
    with pytest.raises(PlotsimpleError, match='You can use: "standard deviation", "standard error", '
                                              '"confidence interval"'):
        ps.chart(marks).bars(x="class", y="marks", error="spread")


@pytest.mark.parametrize("show, how", [("smallest", min), ("biggest", max)])
def test_show_takes_every_measure_name(marks, show, how):
    fig = ps.chart(marks).bars(x="class", y="marks", show=show)._render()
    heights = sorted(p.get_height() for p in fig.axes[0].patches)
    want = sorted(marks.groupby("class")["marks"].agg(how))
    assert heights == pytest.approx(want)
    sd = ps.chart(marks).bars(x="class", y="marks", show="standard deviation")._render()
    got = sorted(p.get_height() for p in sd.axes[0].patches)
    assert got == pytest.approx(sorted(marks.groupby("class")["marks"].std()))
    plt.close("all")


# ---------------------------------------------------------------- numbers
def test_old_number_words_work(capsys):
    nums = ps.numbers([3, None, 5])
    assert nums.count_empty() == 1
    assert nums.done().tolist()[0] == 3
    nums.fill_empty(with_value=0)
    nums.show_first(2)
    ps.number_grid([[1, 2], [3, 4]]).show_last(1)
    assert notes(capsys.readouterr().out) == [
        note('"count_empty"', '"count_empty_numbers"'), note('"done"', '"as_numpy"'),
        note('"fill_empty"', '"fill_empty_numbers"'), note('"show_first"', '"show_first_numbers"'),
        note('"show_last"', '"show_last_rows"')]
