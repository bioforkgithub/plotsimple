"""Tests for the column words: sorting, new columns, combining and reshaping tables."""

import numpy as np
import pandas as pd
import pytest

import plotsimple as ps
from plotsimple import PlotsimpleError
from plotsimple.table import table as make_table

ENGINES = ["pandas", "polars"]


def build(data, engine):
    data = pd.DataFrame(data)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


def plain(t):
    """The table as pandas, with a fresh row index, for comparing."""
    native = t._done()
    if not isinstance(native, pd.DataFrame):
        native = native.to_pandas()
    return native.reset_index(drop=True)


def same_values(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if pd.isna(x) or pd.isna(y):
            assert pd.isna(x) and pd.isna(y), (x, y)
        elif isinstance(x, str) or isinstance(y, str):
            assert x == y
        else:
            assert x == pytest.approx(y), (x, y)


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


@pytest.fixture
def table(engine, marks):
    return build(marks, engine)


# ---------------------------------------------------------------------------- sort_by
def test_sort_by_matches_pandas(table, marks):
    table.sort_by("marks")
    expected = marks.sort_values("marks", kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(plain(table), expected, check_dtype=False)
    table.sort_by("class", "marks", biggest_first=True)
    expected = marks.sort_values(["class", "marks"], ascending=False, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(plain(table), expected, check_dtype=False)


def test_sort_text_and_empty_cells_last(engine):
    t = build({"name": ["Ben", None, "Asha", "Cleo"], "n": [2.0, 1.0, np.nan, 3.0]}, engine)
    t.sort_by("name")
    assert list(plain(t)["name"][:3]) == ["Asha", "Ben", "Cleo"] and pd.isna(plain(t)["name"][3])
    t.sort_by("name", biggest_first=True)
    assert list(plain(t)["name"][:3]) == ["Cleo", "Ben", "Asha"]
    t.sort_by("n", biggest_first=True)
    same_values(plain(t)["n"], [3.0, 2.0, 1.0, np.nan])
    t.sort_by(["n"])
    same_values(plain(t)["n"], [1.0, 2.0, 3.0, np.nan])


def test_sort_ties_keep_order(engine):
    t = build({"k": [1, 1, 1, 0], "order": [1, 2, 3, 4]}, engine)
    t.sort_by("k", biggest_first=True)
    assert list(plain(t)["order"]) == [1, 2, 3, 4]


def test_sort_errors_and_undo(table, marks):
    with pytest.raises(PlotsimpleError, match='Did you mean "marks"'):
        table.sort_by("mark")
    with pytest.raises(PlotsimpleError, match="needs a column"):
        table.sort_by()
    with pytest.raises(PlotsimpleError, match="True or False"):
        table.sort_by("marks", biggest_first="yes")
    table.sort_by("marks")
    table.undo()
    pd.testing.assert_frame_equal(plain(table), marks, check_dtype=False)


# ---------------------------------------------------------------------------- add_column, plain words
def test_add_column_plain_words(table, marks):
    table.add_column("total", by_adding=["marks", "height", "test"])
    same_values(plain(table)["total"], marks["marks"] + marks["height"] + marks["test"])
    table.add_column("diff", by_subtracting=["height", "marks"])
    same_values(plain(table)["diff"], marks["height"] - marks["marks"])
    table.add_column("product", by_multiplying=["marks", "test"])
    same_values(plain(table)["product"], marks["marks"] * marks["test"])
    table.add_column("ratio", by_dividing=["marks", "test"])
    same_values(plain(table)["ratio"], marks["marks"] / marks["test"])
    table.add_column("half", from_column="marks", divide_by=2)
    same_values(plain(table)["half"], marks["marks"] / 2)
    table.add_column("more", from_column="marks", plus=5)
    same_values(plain(table)["more"], marks["marks"] + 5)
    table.add_column("less", from_column="marks", minus="test")
    same_values(plain(table)["less"], marks["marks"] - marks["test"])
    table.add_column("double", from_column="marks", times=2)
    same_values(plain(table)["double"], marks["marks"] * 2)
    table.add_column("school", value="Hill School")
    assert set(plain(table)["school"]) == {"Hill School"}
    table.add_column("year", value=2026)
    assert set(plain(table)["year"]) == {2026}


def test_divide_by_zero_gives_empty_cell(engine):
    t = build({"a": [6.0, 1.0, 0.0], "b": [2.0, 0.0, 0.0], "n": [6, 1, 0], "z": [3, 0, 0]}, engine)
    t.add_column("q", by_dividing=["a", "b"])
    same_values(plain(t)["q"], [3.0, np.nan, np.nan])
    t.add_column("q2", from_column="a", divide_by=0)
    same_values(plain(t)["q2"], [np.nan] * 3)
    t.add_column("q3", "n / z")
    same_values(plain(t)["q3"], [2.0, np.nan, np.nan])
    t.add_column("q4", "n // z")
    same_values(plain(t)["q4"], [2, np.nan, np.nan])
    t.add_column("q5", "n % z")
    same_values(plain(t)["q5"], [0, np.nan, np.nan])
    t.add_column("q6", "sqrt(a - 3)")
    same_values(plain(t)["q6"], [3 ** 0.5, np.nan, np.nan])


def test_add_column_replaces_and_undo(table, marks):
    table.add_column("marks", from_column="marks", plus=1)
    same_values(plain(table)["marks"], marks["marks"] + 1)
    assert list(plain(table).columns) == list(marks.columns)
    table.undo()
    same_values(plain(table)["marks"], marks["marks"])
    table.add_column("x", "marks * 2")
    assert "x" in table.column_names()
    table.undo()
    assert "x" not in table.column_names()


def test_add_column_friendly_errors(table):
    with pytest.raises(PlotsimpleError, match="Choose one way"):
        table.add_column("x")
    with pytest.raises(PlotsimpleError, match="2 ways at once"):
        table.add_column("x", "marks + 1", value=3)
    with pytest.raises(PlotsimpleError, match="needs a name"):
        table.add_column(by_adding=["marks", "test"])
    with pytest.raises(PlotsimpleError, match="two or more column names"):
        table.add_column("x", by_adding="marks")
    with pytest.raises(PlotsimpleError, match='Did you mean "marks"'):
        table.add_column("x", by_adding=["marks", "mrks"])
    with pytest.raises(PlotsimpleError, match="holds text"):
        table.add_column("x", by_adding=["marks", "name"])
    with pytest.raises(PlotsimpleError, match="needs from_column"):
        table.add_column("x", divide_by=2)
    with pytest.raises(PlotsimpleError, match="exactly one of"):
        table.add_column("x", from_column="marks", plus=1, times=2)
    with pytest.raises(PlotsimpleError, match="exactly one of"):
        table.add_column("x", from_column="marks")
    with pytest.raises(PlotsimpleError, match="only works together with from_column"):
        table.add_column("x", value=1, plus=2)
    with pytest.raises(PlotsimpleError, match="number or a column name"):
        table.add_column("x", from_column="marks", plus=[1])
    with pytest.raises(PlotsimpleError, match="has no option called"):
        table.add_column("x", by_add=["marks", "test"])


# ---------------------------------------------------------------------------- add_column, formulas
def test_formulas_match_pandas(table, marks):
    cases = {
        "marks + height": marks["marks"] + marks["height"],
        "marks / 100 * 100": marks["marks"] / 100 * 100,
        "marks / (height / 100) ** 2": marks["marks"] / (marks["height"] / 100) ** 2,
        "-marks + 2 * (test - 1)": -marks["marks"] + 2 * (marks["test"] - 1),
        "marks // 7 + marks % 7": marks["marks"] // 7 + marks["marks"] % 7,
        "round(height, 1)": marks["height"].round(1),
        "round(marks / 3)": (marks["marks"] / 3).round(0),
        "abs(50 - marks)": (50 - marks["marks"]).abs(),
        "sqrt(marks)": np.sqrt(marks["marks"]),
        "log(marks)": np.log(marks["marks"]),
        "log(marks, 10)": np.log10(marks["marks"]),
        "marks ** 2": marks["marks"] ** 2,
        "2 ** -test": 2.0 ** -marks["test"],
        "marks × 2 ÷ 4": marks["marks"] * 2 / 4,
    }
    for i, (formula, expected) in enumerate(cases.items()):
        table.add_column(f"c{i}", formula)
        same_values(plain(table)[f"c{i}"], expected)


def test_formula_backticks_and_keyword_columns(engine):
    t = build({"max temp": [20.0, 25.0], "class": [1, 2], "min temp": [10.0, 15.0]}, engine)
    t.add_column("range", "`max temp` - `min temp`")
    same_values(plain(t)["range"], [10.0, 10.0])
    t.add_column("next", "class + 1")
    same_values(plain(t)["next"], [2, 3])
    with pytest.raises(PlotsimpleError, match='Column "max tmp" not found'):
        t.add_column("x", "`max tmp` * 2")
    with pytest.raises(PlotsimpleError, match="backtick"):
        t.add_column("x", "`max temp * 2")


def test_formula_empty_cells_stay_empty(engine):
    t = build({"a": [1.0, None], "b": [2.0, 3.0]}, engine)
    t.add_column("s", "a + b")
    same_values(plain(t)["s"], [3.0, np.nan])


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os')",
        "__import__('os').system('echo hi')",
        "marks.__class__",
        "marks.real",
        "(lambda: 1)()",
        "lambda x: x",
        "open('x.txt')",
        "eval('1')",
        "exec('1')",
        "[marks]",
        "marks[0]",
        "marks if marks else test",
        "marks > 3",
        "marks == 3",
        "marks and test",
        "'text' + marks",
        "True + marks",
        "marks & 1",
        "marks << 1",
        "~marks",
        "not marks",
        "{marks}",
        "f'{marks}'",
        "(marks := 3)",
        "round(marks, ndigits=1)",
        "round(marks, test)",
        "sqrt(*marks)",
        "globals()",
        "().__class__.__bases__[0].__subclasses__()",
        "marks; import os",
        "import os",
        "x" * 600,
    ],
)
def test_formula_safety(table, formula):
    with pytest.raises(PlotsimpleError):
        table.add_column("x", formula)
    assert "x" not in table.column_names()


def test_formula_friendly_messages(table):
    with pytest.raises(PlotsimpleError, match='Did you mean "marks"') as err:
        table.add_column("x", "mraks + 1")
    assert "Columns you have" in str(err.value)
    with pytest.raises(PlotsimpleError, match=r"\*\* instead of \^"):
        table.add_column("x", "marks ^ 2")
    with pytest.raises(PlotsimpleError, match="A formula can use"):
        table.add_column("x", "__import__('os')")
    with pytest.raises(PlotsimpleError, match=r"Did you mean sqrt\(\)"):
        table.add_column("x", "sqr(marks)")
    with pytest.raises(PlotsimpleError, match="needs brackets"):
        table.add_column("x", "marks + sqrt")
    with pytest.raises(PlotsimpleError, match="no column in it"):
        table.add_column("x", "1 + 2")
    with pytest.raises(PlotsimpleError, match="holds text"):
        table.add_column("x", "name + 1")
    with pytest.raises(PlotsimpleError, match="cannot read"):
        table.add_column("x", "marks + ")
    with pytest.raises(PlotsimpleError, match="base of log"):
        table.add_column("x", "log(marks, 1)")


# ---------------------------------------------------------------------------- group and rank
def test_group_column_boundaries(engine):
    t = build({"marks": [0, 39.9, 40, 59, 60, 79.5, 80, 100, None]}, engine)
    t.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])
    got = list(plain(t)["grade"])
    assert got[:8] == ["D", "D", "C", "C", "B", "B", "A", "A"]
    assert pd.isna(got[8])
    expected = pd.cut(pd.Series([0, 39.9, 40, 59, 60, 79.5, 80, 100]), [-np.inf, 40, 60, 80, np.inf],
                      right=False, labels=["D", "C", "B", "A"])
    assert got[:8] == list(expected.astype(str))
    t.undo()
    assert t.column_names() == ["marks"]


def test_group_column_errors(table):
    with pytest.raises(PlotsimpleError, match="one more name"):
        table.add_group_column("g", from_column="marks", cut_at=[40, 60], names=["low", "high"])
    with pytest.raises(PlotsimpleError, match="small to big"):
        table.add_group_column("g", from_column="marks", cut_at=[60, 40], names=["a", "b", "c"])
    with pytest.raises(PlotsimpleError, match="list of numbers"):
        table.add_group_column("g", from_column="marks", cut_at=["40"], names=["a", "b"])
    with pytest.raises(PlotsimpleError, match="holds text"):
        table.add_group_column("g", from_column="name", cut_at=[40], names=["a", "b"])
    with pytest.raises(PlotsimpleError, match="from_column="):
        table.add_group_column("g", cut_at=[40], names=["a", "b"])
    table.add_group_column("g", from_column="marks", cut_at=50, names=["fail", "pass"])
    assert set(plain(table)["g"]) <= {"fail", "pass"}


def test_rank_column_like_sports_tables(engine):
    t = build({"points": [10, 30, 20, 20, 5]}, engine)
    t.add_rank_column("place", from_column="points")
    assert list(plain(t)["place"]) == [4, 1, 2, 2, 5]
    t.add_rank_column("place_small", from_column="points", biggest_first=False)
    assert list(plain(t)["place_small"]) == [2, 5, 3, 3, 1]
    t.undo()
    assert "place_small" not in t.column_names()


def test_rank_matches_pandas_and_empty_cells(table, marks, engine):
    table.add_rank_column("place", from_column="marks")
    same_values(plain(table)["place"], marks["marks"].rank(method="min", ascending=False))
    t = build({"p": [3.0, None, 1.0]}, engine)
    t.add_rank_column("place", from_column="p")
    same_values(plain(t)["place"], [1, np.nan, 2])
    with pytest.raises(PlotsimpleError, match="from_column="):
        t.add_rank_column("place")
    with pytest.raises(PlotsimpleError, match="not found"):
        t.add_rank_column("place", from_column="q")


# ---------------------------------------------------------------------------- join_with
@pytest.fixture
def people():
    return pd.DataFrame({"name": ["Asha", "Ben", "Zoe"], "age": [12, 13, 11], "height": [1.0, 2.0, 3.0]})


@pytest.mark.parametrize(
    "keep, how",
    [("matching rows", "inner"), ("all rows from this table", "left"), ("all rows from both", "outer")],
)
def test_join_with_matches_pandas(engine, marks, people, keep, how, capsys):
    t = build(marks, engine)
    t.join_with(people, using="name", keep=keep)
    out = capsys.readouterr().out
    assert "12 of 24 rows in this table found a match" in out
    expected = marks.merge(people, on="name", how=how, suffixes=("", "_other"))
    got = plain(t)
    assert sorted(got.columns) == sorted(expected.columns)
    key = ["name", "class", "test"]
    got = got[expected.columns].sort_values(key, na_position="last").reset_index(drop=True)
    expected = expected.sort_values(key, na_position="last").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    t.undo()
    assert len(t) == len(marks) and "age" not in t.column_names()


def test_join_with_any_kind_of_other(engine, marks, people, tmp_path, capsys):
    pl = pytest.importorskip("polars")
    people_file = tmp_path / "people.csv"
    people.to_csv(people_file, index=False)
    others = [people, pl.from_pandas(people), make_table(people), make_table(pl.from_pandas(people)),
              str(people_file), {"name": ["Asha"], "age": [12]}]
    for other in others:
        t = build(marks, engine)
        t.join_with(other, using="name")
        assert "age" in t.column_names()
        assert t._engine == ("Polars" if engine == "polars" else "pandas")
        assert set(plain(t)["name"]) <= {"Asha", "Ben"}


def test_join_with_errors(table, people, capsys):
    with pytest.raises(PlotsimpleError, match="needs using=") as err:
        table.join_with(people)
    assert "Columns in both tables: name, height" in str(err.value)
    with pytest.raises(PlotsimpleError, match='Did you mean keep="matching rows"'):
        table.join_with(people, using="name", keep="matching row")
    with pytest.raises(PlotsimpleError, match='other table has no column "class"'):
        table.join_with(people, using="class")
    with pytest.raises(PlotsimpleError, match="needs another table"):
        table.join_with(using="name")
    with pytest.raises(PlotsimpleError, match="text in this table but numbers"):
        table.join_with({"name": [1, 2]}, using="name")


def test_join_with_repeated_keys_note(table, capsys):
    table.join_with({"name": ["Asha", "Asha"], "pet": ["cat", "dog"]}, using="name")
    out = capsys.readouterr().out
    assert "more than once" in out
    assert len(table) == 12


# ---------------------------------------------------------------------------- add_rows_from
def test_add_rows_from(table, marks, engine):
    pl = pytest.importorskip("polars")
    other = marks.iloc[:3][list(reversed(marks.columns))]  # different column order
    table.add_rows_from(other)
    expected = pd.concat([marks, marks.iloc[:3]], ignore_index=True)
    pd.testing.assert_frame_equal(plain(table), expected, check_dtype=False)
    table.add_rows_from(pl.from_pandas(marks.iloc[:2]))
    assert len(table) == len(marks) + 5
    table.undo()
    table.undo()
    assert len(table) == len(marks)


def test_add_rows_whole_numbers_and_decimals(engine):
    t = build({"a": [1, 2]}, engine)
    t.add_rows_from({"a": [2.5]})
    same_values(plain(t)["a"], [1, 2, 2.5])


def test_add_rows_from_errors(table):
    with pytest.raises(PlotsimpleError) as err:
        table.add_rows_from({"name": ["Asha"], "mark": [3]})
    text = str(err.value)
    assert "Only in this table: class, test, marks, height" in text
    assert "Only in the other table: mark" in text
    with pytest.raises(PlotsimpleError, match="needs another table"):
        table.add_rows_from()


# ---------------------------------------------------------------------------- reshaping
def test_turn_columns_into_rows_matches_pandas(table, marks):
    table.turn_columns_into_rows(["marks", "height"])
    expected = marks.melt(id_vars=["name", "class", "test"], value_vars=["marks", "height"],
                          var_name="column", value_name="value")
    pd.testing.assert_frame_equal(plain(table), expected, check_dtype=False)
    table.undo()
    assert table.column_names() == list(marks.columns)
    table.turn_columns_into_rows("marks", names_to="what", values_to="amount")
    assert table.column_names() == ["name", "class", "test", "height", "what", "amount"]


def test_turn_columns_into_rows_errors(table):
    with pytest.raises(PlotsimpleError, match="needs the columns"):
        table.turn_columns_into_rows()
    with pytest.raises(PlotsimpleError, match='already has a column "name"'):
        table.turn_columns_into_rows(["marks"], names_to="name")
    with pytest.raises(PlotsimpleError, match="all hold numbers or all hold text"):
        table.turn_columns_into_rows(["marks", "name"])
    with pytest.raises(PlotsimpleError, match="not found"):
        table.turn_columns_into_rows(["marks", "weight"])


def test_turn_rows_into_columns_matches_pandas(engine, marks):
    small = marks[["name", "class", "test", "marks"]]
    t = build(small, engine)
    t.turn_rows_into_columns(names_from="test", values_from="marks")
    expected = small.pivot(index=["name", "class"], columns="test", values="marks").reset_index()
    expected.columns = [str(c) for c in expected.columns]
    got = plain(t)
    assert list(got.columns) == ["name", "class", "1", "2", "3"]
    # rows stay in the order they first appear: Asha A, Ben A, ... then the B class
    assert list(got["class"]) == ["A"] * 4 + ["B"] * 4
    got = got.sort_values(["name", "class"]).reset_index(drop=True)
    expected = expected.sort_values(["name", "class"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    t.undo()
    assert len(t) == len(small)


def test_round_trip_long_and_wide(engine):
    weather = ps.example("weather")[["month", "city", "temperature", "rain"]]
    t = build(weather, engine)
    t.turn_columns_into_rows(["temperature", "rain"])
    t.turn_rows_into_columns(names_from="column", values_from="value")
    pd.testing.assert_frame_equal(plain(t), weather, check_dtype=False)


def test_turn_rows_into_columns_errors(table):
    t = build({"name": ["a", "a", "b"], "test": [1, 1, 1], "m": [1, 2, 3]}, "pandas")
    with pytest.raises(PlotsimpleError, match='average_of\\("m", for_each=\\["name", "test"\\]\\)'):
        t.turn_rows_into_columns(names_from="test", values_from="m")
    with pytest.raises(PlotsimpleError, match="needs names_from= and values_from="):
        table.turn_rows_into_columns(names_from="test")
    with pytest.raises(PlotsimpleError, match="not found"):
        table.turn_rows_into_columns(names_from="tests", values_from="marks")


# ---------------------------------------------------------------------------- docstrings
def test_docstring_examples_run():
    for engine in ENGINES:
        marks = build(ps.example("marks"), engine)
        marks.sort_by("marks", biggest_first=True)
        marks.add_column("study_and_marks", by_adding=["hours", "marks"])
        marks.add_column("half", from_column="marks", divide_by=2)
        marks.add_column("school", value="Hill School")
        marks.add_column("marks_per_hour", "marks / hours")
        marks.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])
        marks.add_rank_column("place", from_column="marks")
        marks.join_with({"name": ["Asha", "Ben"], "age": [12, 13]}, using="name")
        marks.add_rows_from(marks)
        weather = build(ps.example("weather"), engine)
        weather.turn_columns_into_rows(["temperature", "rain"])
        weather.undo()
        weather.turn_rows_into_columns(names_from="city", values_from="temperature")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_round_uses_half_to_even_like_r(engine):
    import pandas as pd
    data = {"x": [0.5, 1.5, 2.5, -2.5, 2.675, 1.24, None]}
    if engine == "polars":
        pl = pytest.importorskip("polars")
        t = make_table(pl.DataFrame(data))
    else:
        t = make_table(pd.DataFrame(data))
    t.add_column("r", "round(x)")
    t.add_column("r2", "round(x, 2)")
    t.add_column("r1", "round(x, 1)")
    out = t._done()
    r = [None if v is None or v != v else v for v in list(out["r"])]
    assert r == [0.0, 2.0, 2.0, -2.0, 3.0, 1.0, None]  # same as R round() and numpy.round
    assert list(out["r2"])[:6] == [0.5, 1.5, 2.5, -2.5, 2.68, 1.24]
    assert list(out["r1"])[:6] == [0.5, 1.5, 2.5, -2.5, 2.7, 1.2]
