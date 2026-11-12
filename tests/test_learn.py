"""The learning tools: example tables, ps.words(), ps.help(), ps.lesson() and the cheat sheet."""

import importlib.util
import inspect
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import plotsimple as ps
from plotsimple import learn
from plotsimple.data import EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
KEYS = learn._all_keys()


def _run(code, folder, monkeypatch):
    """Run an example the way a learner would: as a script in an empty folder."""
    monkeypatch.chdir(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # chart.show() with no screen only warns
        exec(compile(code, "<example>", "exec"), {"__name__": "__main__"})
    plt.close("all")


def _saved_files(code):
    return re.findall(r'\.save\("([^"]+)"', code)


# ---------------------------------------------------------------- example tables
@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_tables_load_with_the_columns_described(name):
    table = ps.example(name)
    assert isinstance(table, pd.DataFrame)
    assert len(table) >= 20
    assert list(table.columns) == list(EXAMPLES[name][1])
    assert not table.isna().any().any()


def test_example_without_name_lists_tables(capsys):
    assert ps.example() is None
    out = capsys.readouterr().out
    for name in EXAMPLES:
        assert f'ps.example("{name}")' in out


def test_example_misspelt_says_did_you_mean():
    with pytest.raises(ps.PlotsimpleError, match=r'Did you mean ps.example\("weather"\)'):
        ps.example("wether")


def test_example_accepts_capitals_and_csv_ending():
    assert ps.example("Marks.csv").equals(ps.example("marks"))


def test_example_files_are_inside_the_package():
    folder = ROOT / "src" / "plotsimple" / "data"
    for name in EXAMPLES:
        assert (folder / f"{name}.csv").is_file()


# ---------------------------------------------------------------- the grouping
def _public_ps_functions():
    skip = {"Chart", "NumberGrid", "Numbers", "PlotsimpleError", "Table", "__version__"}
    return [n for n in ps.__all__ if n not in skip]


def test_every_chart_word_is_in_a_group():
    aliases = {a for names in learn._aliases().values() for a in names}
    missing = [w for w in ps.Chart._word_names() if w not in KEYS and w not in aliases]
    assert not missing, f"Add these words to a group in learn.GROUPS: {missing}"


def test_every_table_word_is_in_a_group():
    aliases = {a for key, names in learn._aliases().items() if key.startswith("table.") for a in names}
    missing = [w for w in ps.Table._word_names() if f"table.{w}" not in KEYS and w not in aliases]
    assert not missing, f"Add these table words to a group in learn.GROUPS: {missing}"


def test_every_numbers_word_is_in_a_group():
    missing = [w for w in ps.Numbers._word_names() if f"nums.{w}" not in KEYS]
    assert not missing, f"Add these number words to a group in learn.GROUPS: {missing}"


def test_every_grid_word_is_in_a_group():
    """Grid words that numbers also have (add, average, undo ...) are explained once, as nums words."""
    missing = [w for w in ps.NumberGrid._word_names() if f"grid.{w}" not in KEYS and f"nums.{w}" not in KEYS]
    assert not missing, f"Add these grid words to a group in learn.GROUPS: {missing}"


def test_every_ps_function_is_in_a_group():
    missing = [n for n in _public_ps_functions() if f"ps.{n}" not in KEYS]
    assert not missing, f"Add these to a group in learn.GROUPS: {missing}"


def test_groups_have_no_unknown_or_repeated_words():
    assert len(KEYS) == len(set(KEYS))
    for key in KEYS:
        if key in learn.OPTIONS:
            for word in learn.OPTIONS[key]["words"]:
                params = inspect.signature(getattr(ps.Chart, word)).parameters
                assert key[:-1] in params, f"{word} has no option {key}"
        else:
            assert callable(learn._function(key)), key


def test_every_word_has_a_summary_example_and_help_example():
    for key in KEYS:
        assert learn._summary(key), key
        assert learn._short_example(key), key
        assert key in learn.HELP_EXAMPLES, f"Add a help example for {key}"
    assert set(learn.HELP_EXAMPLES) <= set(KEYS)


def test_aliases_are_found():
    aliases = learn._aliases()
    assert "bar" in aliases["bars"]
    assert "scatter" in aliases["dots"]
    assert "color" in aliases["colour"]


# ---------------------------------------------------------------- ps.words()
def test_words_prints_every_word_in_its_group(capsys):
    ps.words()
    out = capsys.readouterr().out
    for group, keys in learn.GROUPS:
        assert group in out
        for key in keys:
            assert f"  {learn._display(key)} " in out
    assert 'chart.bars(x="name", y="marks")' in out
    assert 'table.keep_rows("marks", above=50)' in out
    assert "(also: bar)" in out


# ---------------------------------------------------------------- ps.help()
@pytest.mark.parametrize("key", KEYS)
def test_help_for_every_word(key, capsys):
    ps.help(key)
    out = capsys.readouterr().out
    assert learn._call_form(key) in out
    assert "Example you can run:" in out
    assert learn.HELP_EXAMPLES[key].splitlines()[-1] in out


@pytest.mark.parametrize("typed", ["bars", "chart.bars", "bars()", ".bars()", "bar", "ps.bars"])
def test_help_understands_ways_of_writing_a_word(typed, capsys):
    ps.help(typed)
    assert capsys.readouterr().out.startswith("chart.bars()")


@pytest.mark.parametrize("typed, expected", [
    ("chart", "ps.chart()"), ("pairs", "ps.pairs()"), ("show=", "show="), ("error", "error="),
    ("color", "chart.colour()"), ("scatter", "chart.dots()"),
])
def test_help_finds_ps_words_options_and_spellings(typed, expected, capsys):
    ps.help(typed)
    assert capsys.readouterr().out.startswith(expected)


def test_help_accepts_the_step_itself(capsys):
    ps.help(ps.Chart.histogram)
    assert capsys.readouterr().out.startswith("chart.histogram()")
    ps.help(ps.Table.keep_rows)
    assert capsys.readouterr().out.startswith("table.keep_rows()")
    ps.help(ps.Table.save)
    out = capsys.readouterr().out
    assert out.startswith("table.save()") and "chart.save()" not in out


@pytest.mark.parametrize("typed", ["keep_rows", "table.keep_rows", "keep_rows()", "table.keep_rows()"])
def test_help_finds_table_words(typed, capsys):
    ps.help(typed)
    assert capsys.readouterr().out.startswith("table.keep_rows()")


def test_help_for_a_word_on_charts_and_tables_explains_both(capsys):
    ps.help("save")
    out = capsys.readouterr().out
    assert out.startswith("chart.save()") and "\ntable.save()\n" in out
    assert 'ps.help("table.save")' in out
    ps.help("table.save")
    out = capsys.readouterr().out
    assert out.startswith("table.save()") and "chart.save()" not in out
    ps.help("chart.save")
    out = capsys.readouterr().out
    assert out.startswith("chart.save()") and "table.save()" not in out


@pytest.mark.parametrize("typed, expected", [
    ("average", "nums.average()"), ("nums.average", "nums.average()"), ("grid.row", "grid.row()"),
    ("swap_rows_and_columns", "grid.swap_rows_and_columns()"), ("numbers", "ps.numbers()"),
])
def test_help_finds_number_words(typed, expected, capsys):
    ps.help(typed)
    assert capsys.readouterr().out.startswith(expected)


def test_help_table_word_options_hide_internal_details(capsys):
    ps.help("keep_rows")
    out = capsys.readouterr().out
    assert "  above=\n" in out and "not given" not in out and "extra" not in out


def test_help_unknown_word_says_did_you_mean(capsys):
    ps.help("histogarm")
    out = capsys.readouterr().out
    assert 'no word called "histogarm"' in out
    assert 'ps.help("histogram")' in out


def test_help_without_word_explains_help(capsys):
    ps.help()
    out = capsys.readouterr().out
    assert 'ps.help("bars")' in out and "ps.words()" in out and "ps.lesson()" in out


def test_help_shows_options_with_defaults(capsys):
    ps.help("bars")
    out = capsys.readouterr().out
    assert 'show=       normally "average"' in out
    assert "sideways=" in out


@pytest.mark.parametrize("key", list(learn.HELP_EXAMPLES))
def test_every_help_example_runs(key, tmp_path, monkeypatch, capsys):
    code = learn.HELP_EXAMPLES[key]
    _run(code, tmp_path, monkeypatch)
    for name in _saved_files(code):
        assert (tmp_path / name).stat().st_size > 0, name


# ---------------------------------------------------------------- docstring examples
def _doc_example_code(key):
    """The word's own short example (from its docstring, as ps.words() and the cheat sheet show
    it), after the lines of its help example that come before it: the example table and, for
    steps such as fit_line, the chart step they go on."""
    word = learn._word(key)
    lines = learn.HELP_EXAMPLES[key].splitlines()
    if key in learn.OPTIONS:
        at = next(i for i, line in enumerate(lines) if key in line)
    else:
        at = next(i for i, line in enumerate(lines) if learn._find_call(line, word, learn._prefixes(key)))
    return "\n".join(lines[:at] + [f"_result = {learn._short_example(key)}"])


@pytest.mark.parametrize("key", [k for k in KEYS if k not in learn.OPTIONS])
def test_every_word_has_an_example_in_its_docstring(key):
    doc = inspect.getdoc(learn._function(key))
    assert learn._find_call(doc, learn._word(key), learn._prefixes(key)), f"Add an example to {key}'s docstring"


@pytest.mark.parametrize("key", KEYS)
def test_docstring_table_is_the_help_example_table(key):
    """A docstring that names its table, (with ps.example("weather")), must match ps.help's example."""
    if key in learn.OPTIONS:
        return
    named = learn._TABLE.search(inspect.getdoc(learn._function(key)) or "")
    if named:
        help_table = learn._TABLE.search(learn.HELP_EXAMPLES[key])
        assert help_table and help_table.group(1) == named.group(1), key


@pytest.mark.parametrize("key", KEYS)
def test_every_docstring_example_runs_and_draws(key, tmp_path, monkeypatch, capsys):
    code = _doc_example_code(key)
    monkeypatch.chdir(tmp_path)
    space = {"__name__": "__main__"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # chart.show() with no screen only warns
        exec(compile(code, f"<{key} example>", "exec"), space)
        result = space["_result"]
        target = result if hasattr(result, "save") else space.get("chart", space.get("page"))
        if target is not None:
            saved = _saved_files(learn.HELP_EXAMPLES[key])
            if any(getattr(target, f"_is_plotsimple_{kind}", False) for kind in ("table", "numbers", "grid")):
                ending = ".csv"  # a table, numbers or a grid saves as a table file
            else:
                ending = Path(saved[-1]).suffix if saved else ".png"
            target.save(f"doc_example{ending}")
            assert (tmp_path / f"doc_example{ending}").stat().st_size > 0
    plt.close("all")


# ---------------------------------------------------------------- ps.lesson()
LESSON_NUMBERS = range(1, 16)


def test_there_are_fifteen_lessons(capsys):
    assert len(learn.LESSONS) == 15
    ps.lesson()
    out = capsys.readouterr().out
    assert "15 lessons" in out
    for i in LESSON_NUMBERS:
        assert f"ps.lesson({i})" in out


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_lesson_prints_all_its_parts(number, capsys):
    ps.lesson(number)
    out = capsys.readouterr().out
    les = learn.LESSONS[number - 1]
    assert les["title"] in out
    for heading in ("In this lesson you will learn", "What each line does", "Try it yourself"):
        assert heading in out
    for _, steps in les["parts"]:
        for code, _ in steps:
            assert f"    {code}" in out


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_every_lesson_example_runs(number, tmp_path, monkeypatch, capsys):
    for code in learn.lesson_code(number):
        _run(code, tmp_path, monkeypatch)
        for name in _saved_files(code):
            assert (tmp_path / name).stat().st_size > 0, name


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_lesson_words_are_real_words(number):
    for word in learn.LESSONS[number - 1]["words"]:
        assert learn._lookup(word) is not None, word


def test_every_group_comes_up_in_a_lesson():
    """The lessons do not need to show every word, but each group must come up."""
    taught = {learn._lookup(w) for les in learn.LESSONS for w in les["words"]}
    for group, keys in learn.GROUPS:
        assert taught & set(keys), f"No lesson teaches anything from {group}"


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_lessons_use_tables_and_charts(number):
    """Every lesson works with separate objects: ps.table, ps.chart or table.chart()."""
    code = "\n".join(learn.lesson_code(number))
    assert any(way in code for way in ("ps.table(", "ps.chart(", ".chart()")), number
    for gone in ("ps.start", "draw_", "story"):
        assert gone not in code, f"lesson {number} uses {gone}"


@pytest.mark.parametrize("bad", [0, 16, "1", 1.0, True])
def test_wrong_lesson_number_is_explained(bad):
    with pytest.raises(ps.PlotsimpleError, match="lessons 1 to 15"):
        ps.lesson(bad)


# ---------------------------------------------------------------- cheat sheet
def test_cheatsheet_is_up_to_date():
    spec = importlib.util.spec_from_file_location("make_cheatsheet", ROOT / "docs" / "make_cheatsheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = (ROOT / "docs" / "cheatsheet.md").read_text(encoding="utf-8")
    assert written == module.make(), "Run: python docs/make_cheatsheet.py"


def test_readme_example_runs(tmp_path, monkeypatch):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    code = text.split("```python", 1)[1].split("```", 1)[0]
    assert 'ps.example("marks")' in code and ".chart()" in code
    _run(code, tmp_path, monkeypatch)


# ---------------------------------------------------------------- old names
OLD_WORDS = {  # old name -> what ps.help shows instead
    "show_first": "table.show_first_rows()",
    "remove_empty_rows": "table.remove_rows_with_empty_cells()",
    "count_empty": "table.count_empty_cells()",
    "fill_empty": "table.fill_empty_cells()",
    "done": "table.as_pandas()",
    "all_points": "chart.all_dots()",
    "colour_bar_label": "chart.label_colour_bar()",
    "one_per": "chart.one_chart_per()",
}


@pytest.mark.parametrize("old, new", list(OLD_WORDS.items()))
def test_help_for_an_old_name_points_to_the_new_one(old, new, capsys):
    ps.help(old)
    out = capsys.readouterr().out
    assert out.startswith(f'"{old}" is now called "{_new_word(new)}"')
    assert f"\n{new}\n" in out  # and the help of the new word follows


def _new_word(call):
    return call.split(".", 1)[1][:-2]


@pytest.mark.parametrize("old, new, where", [
    ("band=", "confidence_band=", "chart.fit_line()"), ("big_data", "fast=", "ps.chart()"),
    ("drawn_as", None, None),
])
def test_help_for_an_old_option_points_to_the_new_one(old, new, where, capsys):
    ps.help(old)
    out = capsys.readouterr().out
    if new is None:  # a new option name is not a word of its own
        assert "no word called" in out
    else:
        assert f'is now called "{new}" in {where}' in out


def test_old_names_are_not_listed(capsys):
    ps.words()
    out = capsys.readouterr().out
    for old in OLD_WORDS:
        assert f"  {old} " not in out and f".{old}(" not in out
    for keys in [k for _, keys in learn.GROUPS for k in keys]:
        assert learn._word(keys) not in OLD_WORDS
    cheat = (ROOT / "docs" / "cheatsheet.md").read_text(encoding="utf-8")
    for old in ("all_points", "colour_bar_label", "remove_empty_rows", "big_data", "one_per(", "x_from",
                ".done()", "show_first(", "band="):
        assert not re.search(r"(?<![a-z_])" + re.escape(old), cheat), old
    lessons = "\n".join(code + explain for les in learn.LESSONS for _, steps in les["parts"]
                        for code, explain in steps)
    for old in ("all_points", "colour_bar_label", "remove_empty_rows", "big_data", "one_per(", "x_from",
                ".done()", "show_first(", "band=", 'error="sd"', 'chart="'):
        assert not re.search(r"(?<![a-z_])" + re.escape(old), lessons), old


def test_new_numbers_words_are_exported():
    assert isinstance(ps.numbers([1, 2]), ps.Numbers)
    assert isinstance(ps.number_grid([[1, 2], [3, 4]]), ps.NumberGrid)
    assert isinstance(ps.random_numbers(how_many=3, seed=1), ps.Numbers)
    assert callable(ps.words) and callable(ps.table)  # functions, not the subpackages
    for name in ("numbers", "number_grid", "random_numbers", "Numbers", "NumberGrid"):
        assert name in ps.__all__
