"""The learning tools: example tables, ps.words(), ps.help(), ps.lesson() and the cheat sheet."""

import importlib.util
import inspect
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
    return [line.split('save("')[1].split('"')[0] for line in code.splitlines() if '.save("' in line]


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
    skip = {"Chart", "PlotsimpleError", "__version__"}
    return [n for n in ps.__all__ if n not in skip]


def test_every_chart_word_is_in_a_group():
    aliases = {a for names in learn._aliases().values() for a in names}
    missing = [w for w in ps.Chart._word_names() if w not in KEYS and w not in aliases]
    assert not missing, f"Add these words to a group in learn.GROUPS: {missing}"


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
            assert f"  {key} " in out
    assert 'chart.bars(x="name", y="marks")' in out
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


# ---------------------------------------------------------------- ps.lesson()
def test_there_are_eight_lessons(capsys):
    assert len(learn.LESSONS) == 8
    ps.lesson()
    out = capsys.readouterr().out
    for i in range(1, 9):
        assert f"ps.lesson({i})" in out


@pytest.mark.parametrize("number", range(1, 9))
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


@pytest.mark.parametrize("number", range(1, 9))
def test_every_lesson_example_runs(number, tmp_path, monkeypatch, capsys):
    for code in learn.lesson_code(number):
        _run(code, tmp_path, monkeypatch)
        for name in _saved_files(code):
            assert (tmp_path / name).stat().st_size > 0, name


@pytest.mark.parametrize("number", range(1, 9))
def test_lesson_words_are_real_words(number):
    for word in learn.LESSONS[number - 1]["words"]:
        assert learn._lookup(word) is not None, word


def test_every_group_comes_up_in_a_lesson():
    """The lessons do not need to show every word, but each group must come up."""
    taught = {learn._lookup(w) for les in learn.LESSONS for w in les["words"]}
    for group, keys in learn.GROUPS:
        assert taught & set(keys), f"No lesson teaches anything from {group}"


@pytest.mark.parametrize("bad", [0, 9, "1", 1.0, True])
def test_wrong_lesson_number_is_explained(bad):
    with pytest.raises(ps.PlotsimpleError, match="lessons 1 to 8"):
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
    assert 'ps.example("marks")' in code
    _run(code, tmp_path, monkeypatch)
