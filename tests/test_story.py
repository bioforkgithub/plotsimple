"""The Story: ps.start(...), one object for table words and chart words, with one history."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import plotsimple as ps


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _story():
    return ps.start(example="marks")


# ---------------------------------------------------------------- starting
def test_start_from_example_file_and_table(tmp_path):
    story = _story()
    assert isinstance(story, ps.Story) and len(story) == len(ps.example("marks"))
    path = tmp_path / "marks.csv"
    ps.example("marks").to_csv(path, index=False)
    assert len(ps.start(file=str(path))) == len(story)
    assert len(ps.start(str(path))) == len(story)  # file= is the first option
    df = ps.example("fruit")
    assert len(ps.start(table=df)) == len(df)
    table = ps.table(df)
    story = ps.start(table=table)
    story.keep_rows("count", above=30)
    assert len(table) == len(df)  # the story never changes your own table


def test_start_with_polars_and_fast():
    pl = pytest.importorskip("polars")
    story = ps.start(table=pl.from_pandas(ps.example("marks")))
    assert isinstance(story.as_polars(), pl.DataFrame)
    assert isinstance(ps.start(example="marks", fast=True).as_polars(), pl.DataFrame)


@pytest.mark.parametrize("kwargs, message", [
    ({}, "needs a table to start from"),
    ({"example": "marks", "file": "x.csv"}, "only one of"),
    ({"example": "marks", "sheet": "a"}, "sheet= is for Excel"),
    ({"table": "marks.xlsx"}, 'ps.start\\(file="marks.xlsx"\\)'),
    ({"example": "marks", "fast": "yes"}, "fast= should be True or False"),
])
def test_start_mistakes_are_explained(kwargs, message):
    with pytest.raises(ps.PlotsimpleError, match=message):
        ps.start(**kwargs)


def test_start_unknown_option_says_did_you_mean():
    with pytest.raises(ps.PlotsimpleError, match="Did you mean example="):
        ps.start(exampel="marks")


def test_start_is_exported():
    assert "start" in ps.__all__ and "Story" in ps.__all__


# ---------------------------------------------------------------- table and chart words
def test_table_words_change_the_story_and_return_it():
    story = _story()
    assert story.keep_rows("marks", above=50) is story
    assert (story.as_pandas()["marks"] > 50).all()
    assert isinstance(story.as_pandas(), pd.DataFrame)
    assert story.column_names() == list(ps.example("marks").columns)


def test_draw_words_and_decorations_go_on_the_current_chart(tmp_path):
    story = _story()
    assert story.draw_bars(x="class", y="marks") is story
    story.title_top("Marks").label_y("Marks").line_at_y(50, label="pass mark")
    chart = story._charts[-1]
    assert [s.name for s in chart._steps][:1] == ["bars"]
    assert {"title_top", "label_y"} <= {s.name for s in chart._steps}
    story.save_chart(tmp_path / "marks.png")
    assert (tmp_path / "marks.png").stat().st_size > 0


@pytest.mark.parametrize("kind, kwargs, table", [
    ("bars", dict(x="class", y="marks"), "marks"), ("lines", dict(x="month", y="rain"), "weather"),
    ("dots", dict(x="hours", y="marks"), "marks"), ("histogram", dict(x="marks"), "marks"),
    ("boxes", dict(x="class", y="marks"), "marks"), ("pie", dict(labels="fruit", values="count"), "fruit"),
    ("heatmap", dict(x="month", y="city", value="rain"), "weather"),
    ("contour", dict(x="east", y="north", z="height"), "island"),
])
def test_every_chart_type_draws(kind, kwargs, table, tmp_path):
    story = ps.start(example=table)
    getattr(story, f"draw_{kind}")(**kwargs)
    story.save_chart(tmp_path / f"{kind}.png")
    assert (tmp_path / f"{kind}.png").stat().st_size > 0


def test_every_chart_type_has_a_draw_word():
    from plotsimple.story import DRAW_WORDS
    from plotsimple.words.charts import ChartWords

    kinds = {n for n in vars(ChartWords) if not n.startswith("_")} - {"bar", "line", "scatter", "box", "violin"}
    assert kinds == set(DRAW_WORDS)
    for kind in DRAW_WORDS:
        assert hasattr(ps.Story, f"draw_{kind}") and not hasattr(ps.Story, kind)


def test_words_have_one_meaning_on_a_story():
    """No word of a story means two things: the only shared word, save, is the story's own."""
    words = set(ps.Story._word_names())
    for clash in ("save", "show", "chart"):
        assert not (clash in words and clash != "save")
    assert {"save", "save_table", "save_chart", "show_chart", "show_first_rows"} <= words


def test_spellings_and_old_options_work():
    story = _story()
    story.draw_dots(x="hours", y="marks", color="class")
    story.fit_line(band=True)
    story.color("blue")
    steps = story._charts[-1]._steps
    assert "class" in str(steps[0].kwargs)
    assert any(s.name == "fit_line" and True in s.kwargs.values() for s in steps)
    assert story._charts[-1]._settings["colour"]


# ---------------------------------------------------------------- the snapshot rule
def test_a_chart_uses_the_table_as_it_was_when_first_drawn():
    story = _story()
    story.draw_bars(x="class", y="marks")
    story.keep_rows("class", equal_to="A")
    first = story._charts[0]
    assert len(first._data) == len(ps.example("marks"))   # later table steps do not change it
    story.new_chart()
    story.draw_dots(x="hours", y="marks")
    assert set(story._charts[1]._data["class"]) == {"A"}  # a new chart uses the table as it is now


def test_a_new_column_in_an_old_chart_explains_the_snapshot_rule():
    story = _story()
    story.draw_dots(x="hours", y="marks")
    story.add_column("double", "marks * 2")
    with pytest.raises(ps.PlotsimpleError, match=r"(?s)as it was when you drew it first.*story\.new_chart\(\)"):
        story.draw_lines(x="hours", y="double")


# ---------------------------------------------------------------- one history
def test_steps_lists_table_and_chart_steps(capsys):
    story = _story()
    story.keep_rows("marks", above=50)
    story.draw_bars(x="class", y="marks")
    story.title_top("Good marks")
    capsys.readouterr()
    story.steps()
    out = capsys.readouterr().out
    assert "1. keep rows where marks is above 50" in out
    assert "2. draw bars of marks for each class" in out
    assert '3. title on top: "Good marks"' in out


def test_undo_takes_back_chart_and_table_steps_in_order(capsys):
    story = _story()
    story.keep_rows("marks", above=50)
    story.draw_bars(x="class", y="marks")
    story.title_top("Good marks")
    story.undo()
    assert "title_top" not in {s.name for s in story._charts[0]._steps}
    story.undo()
    assert story._charts == [] and not story._has_chart()
    assert len(story) < len(ps.example("marks"))
    story.undo()
    assert len(story) == len(ps.example("marks"))
    story.undo()
    assert "Nothing to undo" in capsys.readouterr().out


def test_undo_a_layer_keeps_the_chart():
    story = _story()
    story.draw_boxes(x="class", y="marks")
    story.draw_all_dots(x="class", y="marks")
    story.undo()
    assert [s.name for s in story._charts[0]._steps] == ["boxes"]


def test_undo_size_and_style_puts_settings_back():
    story = _story()
    story.draw_bars(x="class", y="marks")
    story.size(width=4, height=3)
    story.style("dark")
    story.undo()
    story.undo()
    settings = story._charts[0]._settings
    assert settings["width"] == 8.0 and settings["style"] == "paper"


def test_start_again_goes_back_to_the_start_and_can_be_undone():
    story = _story()
    story.keep_rows("marks", above=50)
    story.draw_bars(x="class", y="marks")
    story.start_again()
    assert len(story) == len(ps.example("marks")) and story._charts == []
    story.undo()
    assert len(story) < len(ps.example("marks")) and len(story._charts) == 1


def test_undo_remembers_a_limited_number_of_steps(capsys):
    from plotsimple.table._base import HISTORY_LIMIT

    story = _story()
    for _ in range(HISTORY_LIMIT + 2):
        story.sort_by("marks")
    for _ in range(HISTORY_LIMIT):
        story.undo()
    capsys.readouterr()
    story.undo()
    assert "cannot be taken back" in capsys.readouterr().out
    story.steps()
    assert f"{HISTORY_LIMIT + 2}. " not in capsys.readouterr().out


# ---------------------------------------------------------------- summaries give a new story
def test_summaries_return_a_new_story_that_can_be_drawn(tmp_path, capsys):
    story = ps.start(example="fruit")
    totals = story.total_of("count", for_each="fruit")
    assert "total" in capsys.readouterr().out  # the sentence is still printed
    assert isinstance(totals, ps.Story) and totals is not story
    totals.sort_by("total of count", biggest_first=True)
    totals.draw_bars(x="fruit", y="total of count")
    totals.save_chart(tmp_path / "totals.png")
    assert len(story) == len(ps.example("fruit"))  # the first story is unchanged
    assert isinstance(story.count_rows(), int)     # plain numbers stay numbers


def test_a_test_result_is_a_story():
    result = _story().compare_averages("marks", between="class")
    assert isinstance(result, ps.Story) and len(result) == 1


def test_a_story_can_be_given_to_a_table_word():
    a = ps.start(example="marks")
    b = ps.start(example="marks")
    a.add_rows_from(b)
    assert len(a) == 2 * len(b)


# ---------------------------------------------------------------- saving
def test_save_table_save_chart_and_plain_save(tmp_path):
    story = _story()
    story.draw_bars(x="class", y="marks")
    story.save_table(tmp_path / "t.xlsx").save_table(tmp_path / "t.csv")
    story.save(tmp_path / "plain.csv").save(tmp_path / "plain.png").save(tmp_path / "plain.pdf")
    for name in ("t.xlsx", "t.csv", "plain.csv", "plain.png", "plain.pdf"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert pd.read_csv(tmp_path / "plain.csv").shape == ps.example("marks").shape


@pytest.mark.parametrize("name, message", [
    ("results", "Add a file type"), ("results.doc", 'does not know file type="doc"'),
])
def test_save_mistakes_are_explained(name, message):
    story = _story()
    story.draw_bars(x="class", y="marks")
    with pytest.raises(ps.PlotsimpleError, match=message):
        story.save(name)


def test_save_chart_without_a_chart_is_explained():
    with pytest.raises(ps.PlotsimpleError, match="no chart to save yet"):
        _story().save_chart("x.png")


# ---------------------------------------------------------------- several charts
def test_put_charts_together_with_letters(tmp_path):
    story = _story()
    story.draw_boxes(x="class", y="marks")
    story.new_chart()
    story.draw_dots(x="hours", y="marks")
    story.new_chart()
    story.draw_histogram(x="marks")
    story.put_charts_together(per_row=2, letters=True)
    story.title_top("Figure 1")
    fig = story._page._render()
    letters = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert letters == ["A", "B", "C"]
    assert story._page._settings["title"] == "Figure 1"
    story.save_chart(tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 0


def test_put_charts_together_needs_two_charts():
    story = _story()
    story.draw_bars(x="class", y="marks")
    with pytest.raises(ps.PlotsimpleError, match="at least two charts"):
        story.put_charts_together()


def test_on_a_page_only_page_words_work_and_undo_takes_it_apart():
    story = _story()
    story.draw_bars(x="class", y="marks")
    story.new_chart()
    story.draw_dots(x="hours", y="marks")
    story.put_charts_together()
    with pytest.raises(ps.PlotsimpleError, match="put together on one page"):
        story.label_x("Hours")
    with pytest.raises(ps.PlotsimpleError, match="story.new_chart"):
        story.draw_lines(x="hours", y="marks")
    story.undo()
    story.label_x("Hours")
    assert "label_x" in {s.name for s in story._charts[1]._steps}


def test_new_chart_twice_only_explains(capsys):
    story = _story()
    story.new_chart()
    assert "next draw_ step" in capsys.readouterr().out
    story.draw_bars(x="class", y="marks")
    story.new_chart()
    with pytest.raises(ps.PlotsimpleError, match="Draw a chart first"):
        story.title_top("x")
    story.undo()
    story.title_top("x")  # back on chart 1


def test_grid_takes_each_storys_current_chart_with_letters(tmp_path):
    a, b = _story(), ps.start(example="weather")
    a.draw_bars(x="class", y="marks")
    b.draw_lines(x="month", y="rain")
    page = ps.grid(a, b, per_row=2, letters=True)
    fig = page._render()
    assert [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")] == ["A", "B"]
    page.save(tmp_path / "grid.png")
    with pytest.raises(ps.PlotsimpleError, match="has no chart yet"):
        ps.grid(a, _story())


def test_side_by_side_letters_must_be_true_or_false():
    a = ps.chart(ps.example("marks")).bars(x="class", y="marks")
    with pytest.raises(ps.PlotsimpleError, match="letters=True or letters=False"):
        ps.side_by_side(a, a, letters="yes")


# ---------------------------------------------------------------- friendly errors
def test_a_decoration_before_any_draw_says_draw_a_chart_first():
    with pytest.raises(ps.PlotsimpleError, match=r"Draw a chart first, for example story\.draw_bars\(x=\.\.\., y=\.\.\.\)"):
        _story().title_top("Marks")


@pytest.mark.parametrize("word", ["bars", "scatter", "histogram"])
def test_a_chart_type_without_draw_is_explained(word):
    with pytest.raises(ps.PlotsimpleError, match="start with draw_"):
        getattr(_story(), word)


def test_unknown_word_says_did_you_mean():
    with pytest.raises(ps.PlotsimpleError, match=r"Did you mean \.draw_bars\(\)"):
        _story().draw_bar(x="class", y="marks")
    assert not hasattr(_story(), "nonsense")


def test_show_and_chart_are_explained():
    with pytest.raises(ps.PlotsimpleError, match="show_chart"):
        _story().show()
    with pytest.raises(ps.PlotsimpleError, match="draws its charts itself"):
        _story().chart()


def test_errors_use_story_words():
    story = _story()
    with pytest.raises(ps.PlotsimpleError) as info:
        story.draw_bars(x="class", y="marks", colour=5)
    assert "chart." not in str(info.value).replace("ps.chart", "")


def test_setting_with_equals_is_explained():
    with pytest.raises(ps.PlotsimpleError, match="brackets, not with ="):
        _story().title_top = "x"


def test_unknown_option_names_the_story_word():
    with pytest.raises(ps.PlotsimpleError, match=r"\.draw_bars\(\) has no option called colr="):
        _story().draw_bars(x="class", y="marks", colr="class")


# ---------------------------------------------------------------- showing
def test_repr_and_jupyter():
    story = _story()
    assert "plotsimple story: a table of" in repr(story)
    assert story._repr_png_() is None and story._repr_html_()
    story.draw_bars(x="class", y="marks")
    assert story._repr_html_() is None and story._repr_png_()[:4] == b"\x89PNG"


def test_help_for_a_draw_word_points_to_the_chart_word(capsys):
    ps.help("draw_bars")
    out = capsys.readouterr().out
    assert out.startswith("story.draw_bars()") and 'ps.help("bars")' in out and "chart.bars()" in out
