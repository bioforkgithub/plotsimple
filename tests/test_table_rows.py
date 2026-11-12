"""Tests for the row and column words of tables (rows.py), on pandas and Polars."""

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table


def _engine_table(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


@pytest.fixture(params=["pandas", "polars"])
def engine(request):
    return request.param


@pytest.fixture
def marks_df():
    return sdata.example("marks")


@pytest.fixture
def marks(engine, marks_df):
    return _engine_table(marks_df, engine)


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "name": ["Asha", " Ben  ", "Chen", None, "Dina", "  "],
        "marks": [61.0, np.nan, 45.0, 80.0, 45.0, 70.0],
        "num": ["12.5", "1,200", "12,5", "x", "", None],
        "when": ["2026-03-15", "15/03/2026", "12/25/2026", "15 March 2026", "nope", "2026"],
        "passed": [True, False, True, True, False, True],
    })


@pytest.fixture
def messy(engine, messy_df):
    return _engine_table(messy_df, engine)


def plain(t):
    """The table as plain pandas with a fresh index, whatever the engine."""
    native = t._done()
    if not isinstance(native, pd.DataFrame):
        native = native.to_pandas()
    return native.reset_index(drop=True)


def same(t, expected):
    got = plain(t)
    expected = expected.reset_index(drop=True)
    assert list(got.columns) == list(expected.columns)
    for c in expected.columns:
        assert [None if pd.isna(v) else v for v in got[c]] == [None if pd.isna(v) else v for v in expected[c]], c


# ---------------------------------------------------------------------- looking
def test_describe_columns(marks, capsys):
    assert marks.describe_columns() is None
    out = capsys.readouterr().out
    assert "54 rows and 6 columns" in out
    line = next(l for l in out.splitlines() if l.strip().startswith("marks"))
    df = sdata.example("marks")
    assert "number" in line and str(df["marks"].min()) in line and str(df["marks"].max()) in line
    assert str(round(df["marks"].mean(), 2)) in line
    name_line = next(l for l in out.splitlines() if l.strip().startswith("name"))
    assert "text" in name_line and name_line.split()[-1] == str(df["name"].nunique())


def test_describe_messy(messy, capsys):
    messy.describe_columns()
    out = capsys.readouterr().out
    assert "true/false" in out
    marks_line = next(l for l in out.splitlines() if l.strip().startswith("marks"))
    assert marks_line.split()[2] == "1"  # one empty cell
    name_line = next(l for l in out.splitlines() if l.strip().startswith("name"))
    assert name_line.split()[2] == "2"  # None and "  "
    messy.make_dates("when")
    capsys.readouterr()
    messy.describe_columns()
    when_line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("when"))
    assert "date" in when_line and "2026-01-01" in when_line and "2026-12-25" in when_line


def test_count_empty(messy, marks, capsys):
    messy.count_empty_cells()
    out = capsys.readouterr().out
    import re
    counts = dict(re.findall(r"^  (\w+) +(\d+)$", out, flags=re.M))
    assert counts == {"name": "2", "marks": "1", "num": "2", "when": "0", "passed": "0"}
    assert "Total: 5 cells empty." in out
    marks.count_empty_cells()
    assert "No empty cells." in capsys.readouterr().out


# ---------------------------------------------------------------------- columns
def test_keep_remove_rename_columns(marks, marks_df, capsys):
    marks.keep_columns("marks", "name")
    same(marks, marks_df[["marks", "name"]])
    marks.undo()
    marks.keep_columns(["name", "marks"])
    same(marks, marks_df[["name", "marks"]])
    marks.undo()
    marks.remove_columns("height", "hours")
    same(marks, marks_df.drop(columns=["height", "hours"]))
    marks.rename_column("marks", to="score")
    assert marks.column_names() == ["name", "class", "subject", "score"]
    marks.steps()
    out = capsys.readouterr().out
    assert "remove columns height, hours" in out and 'rename column "marks" to "score"' in out
    marks.undo().undo()
    same(marks, marks_df)


def test_column_errors(marks):
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        marks.keep_columns("mark")
    with pytest.raises(SimplydataError, match="For example"):
        marks.keep_columns()
    with pytest.raises(SimplydataError, match="every column"):
        marks.remove_columns(*marks.column_names())
    with pytest.raises(SimplydataError, match="already a column"):
        marks.rename_column("marks", to="name")
    with pytest.raises(SimplydataError, match="to="):
        marks.rename_column("marks")
    with pytest.raises(SimplydataError, match="no option called new="):
        marks.rename_column("marks", new="score")


# ---------------------------------------------------------------------- choosing rows
CONDITIONS = [
    ("marks", {"above": 50}, lambda d: d["marks"] > 50, "marks is above 50"),
    ("marks", {"below": 50}, lambda d: d["marks"] < 50, "marks is below 50"),
    ("marks", {"at_least": 61}, lambda d: d["marks"] >= 61, "marks is at least 61"),
    ("marks", {"at_most": 61}, lambda d: d["marks"] <= 61, "marks is at most 61"),
    ("hours", {"above": 5.5}, lambda d: d["hours"] > 5.5, "hours is above 5.5"),
    ("class", {"equal_to": "A"}, lambda d: d["class"] == "A", 'class is "A"'),
    ("class", {"not_equal_to": "A"}, lambda d: d["class"] != "A", 'class is not "A"'),
    ("marks", {"equal_to": 61}, lambda d: d["marks"] == 61, "marks is 61"),
    ("marks", {"between": [50, 70]}, lambda d: d["marks"].between(50, 70), "marks is between 50 and 70"),
    ("marks", {"between": [70, 50]}, lambda d: d["marks"].between(50, 70), "marks is between 50 and 70"),
    ("subject", {"one_of": ["Maths", "Science"]}, lambda d: d["subject"].isin(["Maths", "Science"]),
     'subject is one of "Maths", "Science"'),
    ("name", {"contains": "en"}, lambda d: d["name"].str.contains("en", regex=False), 'name contains "en"'),
    ("name", {"starts_with": "A"}, lambda d: d["name"].str.startswith("A"), 'name starts with "A"'),
    ("name", {"ends_with": "a"}, lambda d: d["name"].str.endswith("a"), 'name ends with "a"'),
    ("marks", {"is_empty": True}, lambda d: d["marks"].isna(), "marks is empty"),
    ("marks", {"is_empty": False}, lambda d: d["marks"].notna(), "marks is not empty"),
]


@pytest.mark.parametrize("column, option, mask, words", CONDITIONS)
def test_keep_and_remove_rows(marks, marks_df, column, option, mask, words, capsys):
    expected = marks_df[mask(marks_df)]
    marks.keep_rows(column, **option)
    same(marks, expected)
    out = capsys.readouterr().out
    assert f"Kept {len(expected)} of 54 rows." in out
    marks.steps()
    assert f"keep rows where {words}" in capsys.readouterr().out
    marks.undo()
    same(marks, marks_df)
    marks.remove_rows(column, **option)
    same(marks, marks_df[~mask(marks_df)])
    assert f"Removed {len(expected)} of 54 rows." in capsys.readouterr().out
    marks.undo()
    same(marks, marks_df)


def test_empty_cells_never_match(messy, messy_df):
    messy.keep_rows("marks", below=1000)
    assert len(messy) == 5
    messy.undo()
    messy.remove_rows("marks", above=50)  # the empty mark is not above 50, so it stays
    assert plain(messy)["marks"].isna().sum() == 1
    messy.undo()
    messy.keep_rows("name", is_empty=True)  # None and a cell of spaces
    assert len(messy) == 2
    messy.undo()
    messy.keep_rows("passed", equal_to=True)
    same(messy, messy_df[messy_df["passed"]])


def test_keep_rows_on_dates(messy):
    messy.make_dates("when")
    messy.keep_rows("when", above="2026-03-14")
    assert len(messy) == 4
    messy.undo()
    messy.keep_rows("when", between=["2026-01-01", "2026-03-15"])
    assert len(messy) == 4


def test_row_condition_errors(marks):
    with pytest.raises(SimplydataError, match="needs one condition"):
        marks.keep_rows("marks")
    with pytest.raises(SimplydataError, match="one condition at a time"):
        marks.keep_rows("marks", above=40, below=80)
    with pytest.raises(SimplydataError, match="without quotes: above=50"):
        marks.keep_rows("marks", above="50")
    with pytest.raises(SimplydataError, match="needs a number"):
        marks.remove_rows("marks", equal_to="lots")
    with pytest.raises(SimplydataError, match='needs text in quotes, like equal_to="1"'):
        marks.keep_rows("class", equal_to=1)
    with pytest.raises(SimplydataError, match="compares numbers or dates"):
        marks.keep_rows("name", above=3)
    with pytest.raises(SimplydataError, match="looks inside text"):
        marks.keep_rows("marks", contains="5")
    with pytest.raises(SimplydataError, match="two values"):
        marks.keep_rows("marks", between=50)
    with pytest.raises(SimplydataError, match="list in square brackets"):
        marks.keep_rows("class", one_of="A")
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        marks.keep_rows("mark", above=50)
    with pytest.raises(SimplydataError, match="Did you mean above="):
        marks.keep_rows("marks", abve=50)
    with pytest.raises(SimplydataError, match="one plain word"):
        marks.keep_rows("marks", 50)
    with pytest.raises(SimplydataError, match="True or False"):
        marks.keep_rows("marks", is_empty="yes")
    with pytest.raises(SimplydataError, match=r'sdata.example\("marks"\)'):
        marks.keep_rows("marks")
    assert len(marks) == 54 and not marks._history  # errors never change the table


def test_first_last_random_rows(marks, marks_df, engine, capsys):
    marks.keep_first_rows(10)
    same(marks, marks_df.head(10))
    assert "Kept 10 of 54 rows." in capsys.readouterr().out
    marks.undo()
    marks.keep_last_rows(4)
    same(marks, marks_df.tail(4))
    marks.undo()
    marks.keep_random_rows(10, seed=1)
    first = plain(marks)
    assert len(first) == 10
    assert len(first.merge(marks_df)) == 10  # every row comes from the table
    marks.undo()
    marks.keep_random_rows(10, seed=1)
    assert plain(marks).equals(first)  # same seed, same rows
    marks.undo()
    marks.keep_random_rows(500)
    assert len(marks) == 54
    with pytest.raises(SimplydataError, match="whole number"):
        marks.keep_first_rows(2.5)
    with pytest.raises(SimplydataError, match="seed="):
        marks.keep_random_rows(3, seed="abc")


def test_remove_duplicate_rows(marks, marks_df, capsys):
    marks.remove_duplicate_rows(columns="name")
    same(marks, marks_df.drop_duplicates(subset=["name"]))
    assert f"Removed {54 - marks_df['name'].nunique()} of 54 rows." in capsys.readouterr().out
    marks.undo()
    marks.remove_duplicate_rows(columns=["class", "subject"])
    same(marks, marks_df.drop_duplicates(subset=["class", "subject"]))
    marks.undo()
    marks.keep_columns("class", "subject").remove_duplicate_rows()
    same(marks, marks_df[["class", "subject"]].drop_duplicates())
    with pytest.raises(SimplydataError, match="not found"):
        marks.remove_duplicate_rows(columns=["nope"])


# ---------------------------------------------------------------------- empty cells
def test_remove_empty_rows(messy, messy_df, capsys):
    messy.remove_rows_with_empty_cells()
    same(messy, messy_df.iloc[[0, 2]])
    assert "Removed 4 of 6 rows. 2 rows left." in capsys.readouterr().out
    messy.undo()
    messy.remove_rows_with_empty_cells("marks")
    same(messy, messy_df.drop(index=1))
    messy.undo()
    messy.remove_rows_with_empty_cells("name")
    same(messy, messy_df.drop(index=[3, 5]))


def test_fill_empty(messy, messy_df, capsys):
    filled = messy_df["marks"].fillna(messy_df["marks"].mean())
    messy.fill_empty_cells("marks", with_the="average")
    assert plain(messy)["marks"].tolist() == pytest.approx(filled.tolist())
    assert "Filled 1 cell in marks." in capsys.readouterr().out
    messy.undo()
    messy.fill_empty_cells("marks", with_the="median")
    assert plain(messy)["marks"][1] == messy_df["marks"].median()
    messy.undo()
    messy.fill_empty_cells("marks", with_the="value above")
    assert plain(messy)["marks"].tolist() == messy_df["marks"].ffill().tolist()
    messy.undo()
    messy.fill_empty_cells("marks", with_the="value_below")
    assert plain(messy)["marks"].tolist() == messy_df["marks"].bfill().tolist()
    messy.undo()
    messy.fill_empty_cells("marks", with_value=0)
    assert plain(messy)["marks"].tolist() == messy_df["marks"].fillna(0).tolist()
    messy.undo()
    messy.fill_empty_cells("name", with_value="?")
    assert plain(messy)["name"].tolist() == ["Asha", " Ben  ", "Chen", "?", "Dina", "?"]
    messy.undo()
    messy.fill_empty_cells("name", with_the="value above")
    assert plain(messy)["name"].tolist() == ["Asha", " Ben  ", "Chen", "Chen", "Dina", "Dina"]
    messy.steps()
    assert "fill empty cells in name with the value above" in capsys.readouterr().out


def test_fill_empty_errors(messy):
    with pytest.raises(SimplydataError, match="exactly one"):
        messy.fill_empty_cells("marks")
    with pytest.raises(SimplydataError, match="exactly one"):
        messy.fill_empty_cells("marks", with_value=0, with_the="average")
    with pytest.raises(SimplydataError, match="needs numbers"):
        messy.fill_empty_cells("name", with_the="average")
    with pytest.raises(SimplydataError, match='Did you mean with_the="median"'):
        messy.fill_empty_cells("marks", with_the="medium")
    with pytest.raises(SimplydataError, match="needs text in quotes"):
        messy.fill_empty_cells("name", with_value=0)


# ---------------------------------------------------------------------- fixing columns
def test_make_numbers(messy, capsys):
    messy.make_numbers("num")
    values = plain(messy)["num"].tolist()
    assert values[:3] == [12.5, 1200.0, 12.5]
    assert all(pd.isna(v) for v in values[3:])
    out = capsys.readouterr().out
    assert '1 cell could not become numbers and is now empty, for example: "x"' in out
    messy.make_numbers("passed")
    assert plain(messy)["passed"].tolist() == [1, 0, 1, 1, 0, 1]
    messy.make_numbers("marks")
    assert "already numbers" in capsys.readouterr().out
    messy.undo().undo()
    assert plain(messy)["num"].tolist()[:2] == ["12.5", "1,200"]
    with pytest.raises(SimplydataError, match="not found"):
        messy.make_numbers("nums")


def test_make_dates(messy, capsys):
    messy.make_dates("when")
    got = pd.to_datetime(plain(messy)["when"]).tolist()
    expected = pd.to_datetime(["2026-03-15", "2026-03-15", "2026-12-25", "2026-03-15", None, "2026-01-01"]).tolist()
    assert [None if pd.isna(v) else v for v in got] == [None if pd.isna(v) else v for v in expected]
    assert '"nope"' in capsys.readouterr().out
    messy.undo()
    assert plain(messy)["when"][0] == "2026-03-15"


def test_make_dates_from_years(engine):
    growth = _engine_table(sdata.example("growth"), engine)
    growth.make_dates("year")
    assert pd.Timestamp(plain(growth)["year"][0]) == pd.Timestamp("2000-01-01")
    with pytest.raises(SimplydataError, match="decimals"):
        growth.make_dates("population")


def test_make_text(engine, messy):
    weather = _engine_table(sdata.example("weather"), engine)
    weather.make_text("month_number")
    assert plain(weather)["month_number"].tolist()[:3] == ["1", "2", "3"]
    weather.keep_rows("month_number", equal_to="1")
    assert len(weather) == 3
    messy.make_text("passed")
    assert plain(messy)["passed"].tolist() == ["True", "False", "True", "True", "False", "True"]
    messy.make_dates("when").make_text("when")
    assert plain(messy)["when"].tolist()[0] == "2026-03-15"
    assert pd.isna(plain(messy)["when"][4])  # empty stays empty


def test_text_case_and_spaces(marks, marks_df, messy):
    marks.make_lowercase("name")
    assert plain(marks)["name"].tolist() == marks_df["name"].str.lower().tolist()
    marks.make_uppercase("subject")
    assert plain(marks)["subject"].tolist() == marks_df["subject"].str.upper().tolist()
    marks.undo().undo()
    same(marks, marks_df)
    messy.remove_extra_spaces("name")
    names = plain(messy)["name"].tolist()
    assert names[:3] == ["Asha", "Ben", "Chen"] and names[5] == ""
    with pytest.raises(SimplydataError, match="works on text"):
        marks.make_lowercase("marks")


def test_replace_values(marks, marks_df, capsys):
    marks.replace_values("subject", "Maths", with_value="Math")
    assert plain(marks)["subject"].tolist() == marks_df["subject"].replace("Maths", "Math").tolist()
    assert f"Replaced {(marks_df['subject'] == 'Maths').sum()} cells in subject." in capsys.readouterr().out
    marks.undo()
    marks.replace_values("class", ["A", "B"], with_value="AB")
    assert plain(marks)["class"].tolist() == marks_df["class"].replace(["A", "B"], "AB").tolist()
    marks.undo()
    marks.replace_values("marks", 61, with_value=None)
    assert plain(marks)["marks"].isna().sum() == (marks_df["marks"] == 61).sum()
    marks.steps()
    assert 'replace 61 in marks with empty' in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="with_value="):
        marks.replace_values("subject", "Maths")
    with pytest.raises(SimplydataError, match="needs a number"):
        marks.replace_values("hours", 6.3, with_value="six")


# ---------------------------------------------------------------------- undo after every kind of step
STEPS = [
    lambda t: t.keep_columns("name"),
    lambda t: t.remove_columns("marks"),
    lambda t: t.rename_column("marks", to="score"),
    lambda t: t.keep_rows("marks", above=50),
    lambda t: t.remove_rows("name", is_empty=True),
    lambda t: t.keep_first_rows(2),
    lambda t: t.keep_last_rows(2),
    lambda t: t.keep_random_rows(2, seed=3),
    lambda t: t.remove_duplicate_rows(columns="marks"),
    lambda t: t.remove_rows_with_empty_cells(),
    lambda t: t.fill_empty_cells("marks", with_the="average"),
    lambda t: t.make_numbers("num"),
    lambda t: t.make_dates("when"),
    lambda t: t.make_text("marks"),
    lambda t: t.make_lowercase("name"),
    lambda t: t.make_uppercase("name"),
    lambda t: t.remove_extra_spaces("name"),
    lambda t: t.replace_values("name", "Asha", with_value="Asa"),
]


@pytest.mark.parametrize("step", STEPS)
def test_undo_every_step(messy, messy_df, step):
    assert step(messy) is messy
    assert len(messy._history) == 1
    messy.undo()
    same(messy, messy_df)
