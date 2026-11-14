"""The learning tools: example tables, sdata.words(), sdata.help(), sdata.lesson() and the cheat sheet."""

import importlib.util
import inspect
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import learn
from simplydata.data import EXAMPLES

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
    table = sdata.example(name)
    assert isinstance(table, pd.DataFrame)
    assert len(table) >= 20
    assert list(table.columns) == list(EXAMPLES[name][1])
    if name == "shop":  # the one table made for practising empty cells and dates
        assert table.isna().any().any() and pd.api.types.is_datetime64_any_dtype(table["date"])
    elif name == "school":  # empty sleep cells, for multiple imputation
        assert table["sleep"].isna().any() and not table.drop(columns="sleep").isna().any().any()
    else:
        assert not table.isna().any().any()


def test_example_without_name_lists_tables(capsys):
    assert sdata.example() is None
    out = capsys.readouterr().out
    for name in EXAMPLES:
        assert f'sdata.example("{name}")' in out


def test_example_misspelt_says_did_you_mean():
    with pytest.raises(sdata.SimplydataError, match=r'Did you mean sdata.example\("weather"\)'):
        sdata.example("wether")


def test_example_accepts_capitals_and_csv_ending():
    assert sdata.example("Marks.csv").equals(sdata.example("marks"))


def test_example_files_are_inside_the_package():
    folder = ROOT / "src" / "simplydata" / "data"
    for name in EXAMPLES:
        assert (folder / f"{name}.csv").is_file()


# ---------------------------------------------------------------- the grouping
def _public_ps_functions():
    skip = {"Chart", "NumberGrid", "Numbers", "SimplydataError", "Table", "__version__"}
    return [n for n in sdata.__all__ if n not in skip]


def test_every_chart_word_is_in_a_group():
    aliases = {a for names in learn._aliases().values() for a in names}
    missing = [w for w in sdata.Chart._word_names() if w not in KEYS and w not in aliases]
    assert not missing, f"Add these words to a group in learn.GROUPS: {missing}"


def test_every_table_word_is_in_a_group():
    aliases = {a for key, names in learn._aliases().items() if key.startswith("table.") for a in names}
    missing = [w for w in sdata.Table._word_names() if f"table.{w}" not in KEYS and w not in aliases]
    assert not missing, f"Add these table words to a group in learn.GROUPS: {missing}"


def test_every_numbers_word_is_in_a_group():
    missing = [w for w in sdata.Numbers._word_names() if f"nums.{w}" not in KEYS]
    assert not missing, f"Add these number words to a group in learn.GROUPS: {missing}"


def test_every_grid_word_is_in_a_group():
    """Grid words that numbers also have (add, average, undo ...) are explained once, as nums words."""
    missing = [w for w in sdata.NumberGrid._word_names() if f"grid.{w}" not in KEYS and f"nums.{w}" not in KEYS]
    assert not missing, f"Add these grid words to a group in learn.GROUPS: {missing}"


def test_every_distribution_word_is_in_a_group():
    from simplydata.distributions import Distribution

    missing = [w for w in Distribution._word_names() if f"dist.{w}" not in KEYS]
    assert not missing, f"Add these distribution words to a group in learn.GROUPS: {missing}"


def test_every_ps_function_is_in_a_group():
    missing = [n for n in _public_ps_functions() if f"sdata.{n}" not in KEYS]
    assert not missing, f"Add these to a group in learn.GROUPS: {missing}"


def test_groups_have_no_unknown_or_repeated_words():
    assert len(KEYS) == len(set(KEYS))
    for key in KEYS:
        if key in learn.OPTIONS:
            for word in learn.OPTIONS[key]["words"]:
                params = inspect.signature(getattr(sdata.Chart, word)).parameters
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


# ---------------------------------------------------------------- sdata.words()
def test_words_prints_every_word_in_its_group(capsys):
    sdata.words()
    out = capsys.readouterr().out
    for group, keys in learn.GROUPS:
        assert group in out
        for key in keys:
            assert f"  {learn._display(key)} " in out
    assert 'chart.bars(x="name", y="marks")' in out
    assert 'table.keep_rows("marks", above=50)' in out
    assert "(also: bar)" in out


# ---------------------------------------------------------------- sdata.help()
@pytest.mark.parametrize("key", KEYS)
def test_help_for_every_word(key, capsys):
    sdata.help(key)
    out = capsys.readouterr().out
    assert learn._call_form(key) in out
    assert "Example you can run:" in out
    assert learn.HELP_EXAMPLES[key].splitlines()[-1] in out


@pytest.mark.parametrize("typed", ["bars", "chart.bars", "bars()", ".bars()", "bar", "sdata.bars"])
def test_help_understands_ways_of_writing_a_word(typed, capsys):
    sdata.help(typed)
    assert capsys.readouterr().out.startswith("chart.bars()")


@pytest.mark.parametrize("typed, expected", [
    ("chart", "sdata.chart()"), ("pairs", "sdata.pairs()"), ("show=", "show="), ("error", "error="),
    ("color", "chart.colour()"), ("scatter", "chart.dots()"),
])
def test_help_finds_ps_words_options_and_spellings(typed, expected, capsys):
    sdata.help(typed)
    assert capsys.readouterr().out.startswith(expected)


def test_help_accepts_the_step_itself(capsys):
    sdata.help(sdata.Chart.histogram)
    assert capsys.readouterr().out.startswith("chart.histogram()")
    sdata.help(sdata.Table.keep_rows)
    assert capsys.readouterr().out.startswith("table.keep_rows()")
    sdata.help(sdata.Table.save)
    out = capsys.readouterr().out
    assert out.startswith("table.save()") and "chart.save()" not in out


@pytest.mark.parametrize("typed", ["keep_rows", "table.keep_rows", "keep_rows()", "table.keep_rows()"])
def test_help_finds_table_words(typed, capsys):
    sdata.help(typed)
    assert capsys.readouterr().out.startswith("table.keep_rows()")


def test_help_for_a_word_on_charts_and_tables_explains_both(capsys):
    sdata.help("save")
    out = capsys.readouterr().out
    assert out.startswith("chart.save()") and "\ntable.save()\n" in out
    assert 'sdata.help("table.save")' in out
    sdata.help("table.save")
    out = capsys.readouterr().out
    assert out.startswith("table.save()") and "chart.save()" not in out
    sdata.help("chart.save")
    out = capsys.readouterr().out
    assert out.startswith("chart.save()") and "table.save()" not in out


@pytest.mark.parametrize("typed, expected", [
    ("average", "nums.average()"), ("nums.average", "nums.average()"), ("grid.row", "grid.row()"),
    ("swap_rows_and_columns", "grid.swap_rows_and_columns()"), ("numbers", "sdata.numbers()"),
])
def test_help_finds_number_words(typed, expected, capsys):
    sdata.help(typed)
    assert capsys.readouterr().out.startswith(expected)


def test_help_table_word_options_hide_internal_details(capsys):
    sdata.help("keep_rows")
    out = capsys.readouterr().out
    assert "  above=\n" in out and "not given" not in out and "extra" not in out


def test_help_unknown_word_says_did_you_mean(capsys):
    sdata.help("histogarm")
    out = capsys.readouterr().out
    assert 'no word called "histogarm"' in out
    assert 'sdata.help("histogram")' in out


def test_help_without_word_explains_help(capsys):
    sdata.help()
    out = capsys.readouterr().out
    assert 'sdata.help("bars")' in out and "sdata.words()" in out and "sdata.lesson()" in out


def test_help_shows_options_with_defaults(capsys):
    sdata.help("bars")
    out = capsys.readouterr().out
    assert re.search(r'show=\s+normally "average"', out)
    assert "sideways=" in out


@pytest.mark.parametrize("key", list(learn.HELP_EXAMPLES))
def test_every_help_example_runs(key, tmp_path, monkeypatch, capsys):
    code = learn.HELP_EXAMPLES[key]
    _run(code, tmp_path, monkeypatch)
    for name in _saved_files(code):
        assert (tmp_path / name).stat().st_size > 0, name


# ---------------------------------------------------------------- docstring examples
def _doc_example_code(key):
    """The word's own short example (from its docstring, as sdata.words() and the cheat sheet show
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
    """A docstring that names its table, (with sdata.example("weather")), must match sdata.help's example."""
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
            if any(getattr(target, f"_is_simplydata_{kind}", False) for kind in ("table", "numbers", "grid")):
                ending = ".csv"  # a table, numbers or a grid saves as a table file
            else:
                ending = Path(saved[-1]).suffix if saved else ".png"
            target.save(f"doc_example{ending}")
            assert (tmp_path / f"doc_example{ending}").stat().st_size > 0
    plt.close("all")


# ---------------------------------------------------------------- sdata.lesson()
LESSON_NUMBERS = range(1, 21)


def test_there_are_twenty_lessons(capsys):
    assert len(learn.LESSONS) == 20
    sdata.lesson()
    out = capsys.readouterr().out
    assert "20 lessons" in out
    for i in LESSON_NUMBERS:
        assert f"sdata.lesson({i})" in out


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_lesson_prints_all_its_parts(number, capsys):
    sdata.lesson(number)
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
    """Every lesson works with separate objects: sdata.table, sdata.chart or table.chart()."""
    code = "\n".join(learn.lesson_code(number))
    assert any(way in code for way in ("sdata.table(", "sdata.chart(", ".chart()")), number
    for gone in ("sdata.start", "draw_", "story"):
        assert gone not in code, f"lesson {number} uses {gone}"


@pytest.mark.parametrize("bad", [0, 21, "1", 1.0, True])
def test_wrong_lesson_number_is_explained(bad):
    with pytest.raises(sdata.SimplydataError, match="lessons 1 to 20"):
        sdata.lesson(bad)


# ---------------------------------------------------------------- cheat sheet
def test_cheatsheet_is_up_to_date():
    spec = importlib.util.spec_from_file_location("make_cheatsheet", ROOT / "docs" / "make_cheatsheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = (ROOT / "docs" / "cheatsheet.md").read_text(encoding="utf-8")
    assert written == module.make(), "Run: python docs/make_cheatsheet.py"


# ---------------------------------------------------------------- solutions and the manual
SOLUTIONS = [(n, t) for n in LESSON_NUMBERS for t in range(1, len(learn.LESSONS[n - 1]["task"]) + 1)]


@pytest.mark.parametrize("number", LESSON_NUMBERS)
def test_every_task_has_a_worked_solution(number):
    les = learn.LESSONS[number - 1]
    solutions = learn.lesson_solutions(number)
    assert len(solutions) == len(les["task"]), f"lesson {number}: one solution for each task"
    for source, explain in solutions:
        assert source.startswith("import simplydata as sdata\n"), "a solution runs on its own"
        assert explain.strip() and "\n" not in explain


@pytest.mark.parametrize("number, task", SOLUTIONS)
def test_every_solution_runs(number, task, tmp_path, monkeypatch, capsys):
    source, _ = learn.lesson_solutions(number)[task - 1]
    _run(source, tmp_path, monkeypatch)
    for name in _saved_files(source):
        assert (tmp_path / name).stat().st_size > 0, name


def test_lesson_shows_solutions_only_when_asked(capsys):
    sdata.lesson(3)
    out = capsys.readouterr().out
    assert "Solutions" not in out and "sdata.lesson(3, solutions=True)" in out
    sdata.lesson(3, solutions=True)
    out = capsys.readouterr().out
    assert "\nSolutions\n" in out
    for source, _ in learn.lesson_solutions(3):
        for line in source.splitlines():
            assert f"        {line}" in out


def _make_manual():
    spec = importlib.util.spec_from_file_location("make_manual", ROOT / "docs" / "make_manual.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_is_up_to_date():
    """The manual's LaTeX is made from the code; it must match what the code makes now."""
    written = (ROOT / "docs" / "manual" / "simplydata_manual.tex").read_text(encoding="utf-8")
    assert written == _make_manual().make(), "Run: python docs/make_manual.py"


def test_manual_has_every_figure_and_word():
    manual = _make_manual()
    tex = (ROOT / "docs" / "manual" / "simplydata_manual.tex").read_text(encoding="utf-8")
    for figure in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
        assert (ROOT / "docs" / "manual" / figure).is_file(), f"{figure} is missing: run python docs/make_manual.py"
    entries = set(re.findall(r"^\\wordentry\{.*\}\{(word:[^}]+)\}$", tex, flags=re.M))
    assert entries == {manual.label(key) for key in KEYS}, "every word has one dictionary entry"
    for number in LESSON_NUMBERS:
        assert re.search(rf"^\\unsection\{{Lesson {number}: .*\}}\{{solutions:{number}\}}$", tex, flags=re.M), number


def _calls_in(latex):
    """(code, word, options) for each call written by hand in \\code{...}."""
    for code in re.findall(r"\\code\{((?:[^{}]|\{[^{}]*\})*)\}", latex):
        code = code.replace("\\_", "_").replace("\\%", "%")
        bare = re.sub(r'"[^"]*"', '""', code)  # formulas inside quotes are not calls
        for m in re.finditer(r"\b([a-z_]\w*)\(", bare):
            depth, i = 1, m.end()
            while i < len(bare) and depth:
                depth += {"(": 1, ")": -1}.get(bare[i], 0)
                i += 1
            args = bare[m.end():i - 1]
            while True:  # only the options of this call, not of calls inside it
                shorter = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "", args)
                if shorter == args:
                    break
                args = shorter
            yield code, m.group(1), re.findall(r"\b(\w+)\s*=", args)


def test_hand_written_manual_text_uses_current_words_and_options():
    """The ideas chapters and the introduction are typed by hand: their words and options must
    be today's names (an old name still works for one release, but the manual teaches the new one)."""
    texts = [(ROOT / "docs" / "manual" / "ideas.tex").read_text(encoding="utf-8"),
             _make_manual().GETTING_STARTED]
    keys = learn._all_keys()
    checked = 0
    for text in texts:
        for code, word, options in _calls_in(text):
            found = [k for k in keys if learn._word(k) == word]
            assert found, f"{word}() in \\code{{{code}}} is not a simplydata word"
            names = set()
            for key in found:
                names |= set(inspect.signature(learn._function(key)).parameters)
            unknown = [o for o in options if o not in names]
            assert not unknown, f"\\code{{{code}}}: {word}() has no option {unknown}"
            checked += 1
    assert checked > 20


def test_lesson_count_is_not_typed_by_hand():
    tex = (ROOT / "docs" / "manual" / "simplydata_manual.tex").read_text(encoding="utf-8")
    words = _make_manual().in_words(len(learn.LESSONS))
    assert f"{words.capitalize()} short lessons" in tex
    assert f"There are {words} short lessons" in (ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- the dictionary
DICTIONARY =ROOT / "docs" / "manual" / "simplydataDictionary.tex"


def _make_dictionary():
    spec = importlib.util.spec_from_file_location("make_dictionary", ROOT / "docs" / "make_dictionary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dictionary_is_up_to_date():
    """The dictionary's LaTeX is made from the code (with the manual's page numbers written in it)."""
    dictionary = _make_dictionary()
    written = DICTIONARY.read_text(encoding="utf-8")
    pages = dictionary.pages_in(written)
    assert written == dictionary.make(pages, dictionary.fingerprint_in(written)), \
        "Run: python docs/make_manual.py --no-run"
    assert (ROOT / "docs" / "manual" / "simplydataDictionary.pdf").is_file()


def test_dictionary_pages_come_from_this_manual():
    """The page numbers were read from the manual as it is now, so they match its pages."""
    dictionary = _make_dictionary()
    written = DICTIONARY.read_text(encoding="utf-8")
    assert dictionary.fingerprint_in(written) == dictionary.manual_fingerprint(), \
        "The manual changed: run python docs/make_manual.py --no-run to rebuild both"


def test_dictionary_has_every_word_with_a_page():
    dictionary, manual = _make_dictionary(), _make_manual()
    written = DICTIONARY.read_text(encoding="utf-8")
    pages = dictionary.pages_in(written)
    heads = set(re.findall(r"^\\headword\{.*\}\{(dict:[^}]+)\}$", written, flags=re.M))
    for key in KEYS:
        assert dictionary.dict_label(dictionary.headword(key)) in heads, f"{key} has no entry"
        page, anchor = pages.get(manual.label(key), ("", ""))
        assert page.isdigit() and anchor, f"{key} has no manual page"
    for number in LESSON_NUMBERS:
        assert pages[f"lesson:{number}"][0].isdigit(), f"lesson {number} has no page"
    # each part of an entry ends with its manual page
    entries = re.split(r"^\\headword", written, flags=re.M)[1:]
    assert len(entries) == len(dictionary.entries())
    for entry in entries:
        assert entry.count(r"\sense{") == entry.count(r"\where{Manual \manualpage{word:")
    lesson = next(n for n in LESSON_NUMBERS if "table.keep_rows" in learn.LESSONS[n - 1]["words"])
    assert re.search(r"\\where\{Manual \\manualpage\{word:table\.keep-rows\}\{\d+\}\{[^}]+\}; "
                     rf"lesson {lesson} ", written)


def test_dictionary_lists_every_other_name_and_old_name():
    from simplydata._other_names import OTHER_NAMES

    dictionary = _make_dictionary()
    written = DICTIONARY.read_text(encoding="utf-8")
    listed = {re.sub(r"\\(.)", r"\1", name) for name in re.findall(r"^\\othername\{(.*?)\}\{", written, flags=re.M)}
    for kind, names in OTHER_NAMES.items():
        prefix = {"numbers": "nums"}.get(kind, kind)
        for name in names:
            assert name in listed or f"{prefix}.{name}" in listed, name
    for old, _, _ in dictionary.old_names():
        assert old in listed, old
    for old in ("fit_straight_line", "all_points", "band=", "big_data="):
        assert old in listed, old


@pytest.mark.parametrize("text, expected", [
    ("50 % & $5 #1 {a} ~ ^ \\ _", r"50 \% \& \$5 \#1 \{a\} \textasciitilde{} \textasciicircum{} \textbackslash{} \_"),
    ("20 °C, R²", r"20 \textdegree{}C, R\textsuperscript{2}"),
])
def test_manual_escapes_latex_special_characters(text, expected):
    assert _make_manual().esc(text) == expected


def test_readme_example_runs(tmp_path, monkeypatch):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    code = text.split("```python", 1)[1].split("```", 1)[0]
    assert 'sdata.example("marks")' in code and ".chart()" in code
    _run(code, tmp_path, monkeypatch)


# ---------------------------------------------------------------- old names
OLD_WORDS = {  # old name -> what sdata.help shows instead
    "fit_straight_line": "table.fit_line()",
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
    sdata.help(old)
    out = capsys.readouterr().out
    assert out.startswith(f'"{old}" is now called "{_new_word(new)}"')
    assert f"\n{new}\n" in out  # and the help of the new word follows


def _new_word(call):
    return call.split(".", 1)[1][:-2]


@pytest.mark.parametrize("old, new, where", [
    ("band=", "confidence_band=", "chart.fit_line()"), ("big_data", "fast=", "sdata.chart()"),
    ("drawn_as", None, None),
])
def test_help_for_an_old_option_points_to_the_new_one(old, new, where, capsys):
    sdata.help(old)
    out = capsys.readouterr().out
    if new is None:  # a new option name is not a word of its own
        assert "no word called" in out
    else:
        assert f'is now called "{new}" in {where}' in out


def test_old_names_are_not_listed(capsys):
    sdata.words()
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
    assert isinstance(sdata.numbers([1, 2]), sdata.Numbers)
    assert isinstance(sdata.number_grid([[1, 2], [3, 4]]), sdata.NumberGrid)
    assert isinstance(sdata.random_numbers(how_many=3, seed=1), sdata.Numbers)
    assert callable(sdata.words) and callable(sdata.table)  # functions, not the subpackages
    for name in ("numbers", "number_grid", "random_numbers", "Numbers", "NumberGrid"):
        assert name in sdata.__all__
