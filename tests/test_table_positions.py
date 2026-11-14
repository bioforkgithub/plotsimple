"""Choosing parts of a table by position, one row or cell, reverse_rows, keep_columns by run/kind/name,
replace_values with conditions, the printed row numbers (from 1) and the friendly pointers for [ ] and dots.

Every result is checked against the pandas .iloc / Polars expression written next to it."""

import datetime as dt
import re

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
def weather_df():
    return sdata.example("weather")


@pytest.fixture
def weather(engine, weather_df):
    return _engine_table(weather_df, engine)


def plain(t):
    native = t._done()
    if not isinstance(native, pd.DataFrame):
        native = native.to_pandas()
    return native.reset_index(drop=True)


def same(t, expected):
    pd.testing.assert_frame_equal(plain(t), expected.reset_index(drop=True), check_dtype=False)


# ====================================================================== P0: row numbers from 1
def _labels(out):
    """The row numbers printed at the start of each row of a printed table."""
    return [int(line.split()[0]) for line in out.splitlines() if re.match(r"^\d+\s", line)]


def test_show_last_rows_numbers_rows_in_the_whole_table(marks, capsys):
    marks.show_last_rows(3)
    out = capsys.readouterr().out
    assert _labels(out) == [52, 53, 54]
    assert not re.search(r"^51 ", out, re.M)


def test_show_first_rows_and_show_count_from_1(marks, capsys):
    marks.show_first_rows(3)
    assert _labels(capsys.readouterr().out) == [1, 2, 3]
    marks.show(rows=3)
    assert _labels(capsys.readouterr().out) == [1, 2, 3]
    marks.keep_rows("class", equal_to="B")
    capsys.readouterr()
    marks.show()
    labels = _labels(capsys.readouterr().out)
    assert labels == list(range(1, len(marks) + 1))  # numbered again after a step


def test_repr_and_notebook_display_count_from_1(marks):
    text = repr(marks)
    assert _labels(text)[:3] == [1, 2, 3]
    html = marks._repr_html_()
    assert "<th>1</th>" in html and "<th>54</th>" in html and "<th>0</th>" not in html


def test_big_polars_table_in_a_notebook_shows_both_ends():
    pl = pytest.importorskip("polars")
    t = make_table(pl.DataFrame({"x": range(20_000)}))
    html = t._repr_html_()
    assert "<th>1</th>" in html and "<th>20000</th>" in html and "<th>0</th>" not in html


def test_stored_table_keeps_its_own_index(marks):
    marks.show_last_rows(3)
    marks.show()
    repr(marks)
    marks._repr_html_()
    assert marks.as_pandas().index[0] == 0
    assert list(marks.as_pandas().index[:3]) == [0, 1, 2]


def test_show_rows_with_outliers_gives_row_numbers_in_the_whole_table(engine, capsys):
    data = pd.DataFrame({"x": [10.0, 11, 12, 10, 11, 12, 10, 11, 100, 12, 11, -50]})
    t = _engine_table(data, engine)
    t.show_rows_with_outliers("x")
    out = capsys.readouterr().out
    expected = [i + 1 for i in np.flatnonzero(data["x"].isin([100, -50]))]  # pandas: positions of the outliers
    assert _labels(out) == expected == [9, 12]
    assert "table.remove_rows_by_position([9, 12])" in out
    t.remove_rows_by_position([9, 12])
    assert plain(t)["x"].tolist() == data["x"].drop(index=data.index[[8, 11]]).tolist()


# ====================================================================== keep_rows_by_position
def test_keep_a_run_of_rows(marks, marks_df, capsys):
    marks.keep_rows_by_position(10, to=20)
    assert "Kept rows 10 to 20 (11 of 54 rows)." in capsys.readouterr().out
    same(marks, marks_df.iloc[9:20])  # pandas df.iloc[9:20]


def test_keep_every_nth_row(marks, marks_df, capsys):
    marks.keep_rows_by_position(1, step=10)
    out = capsys.readouterr().out
    assert "Kept every 10th row from row 1 to row 54: rows 1, 11, 21, 31, 41, 51 (6 of 54 rows)." in out
    same(marks, marks_df.iloc[0::10])


def test_every_nth_row_with_an_end(marks, marks_df, capsys):
    marks.keep_rows_by_position(3, to=20, step=3)
    assert "every 3rd row from row 3 to row 20" in capsys.readouterr().out
    same(marks, marks_df.iloc[2:20:3])


def test_keep_a_list_of_rows_in_that_order(marks, marks_df, capsys):
    marks.keep_rows_by_position([9, 1, 5])
    assert "Kept rows 9, 1, 5, in that order (3 of 54 rows)." in capsys.readouterr().out
    same(marks, marks_df.take([8, 0, 4]))  # pandas df.take / Polars df[[8, 0, 4]]


def test_a_range_is_a_list(marks, marks_df):
    marks.keep_rows_by_position(range(1, 6))
    same(marks, marks_df.iloc[0:5])


def test_long_lists_are_printed_short(marks, capsys):
    marks.keep_rows_by_position(list(range(1, 15)))
    out = capsys.readouterr().out
    assert "rows 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., in that order (14 of 54 rows)" in out


def test_keep_one_row_and_the_last_row(marks, marks_df, capsys):
    marks.keep_rows_by_position(3)
    assert "Kept row 3 (1 of 54 rows)." in capsys.readouterr().out
    same(marks, marks_df.iloc[[2]])
    marks.undo()
    marks.keep_rows_by_position("last")
    same(marks, marks_df.iloc[[-1]])
    marks.undo()
    marks.keep_rows_by_position(50, to="last")
    same(marks, marks_df.iloc[49:])


def test_to_beyond_the_end_stops_at_the_last_row_with_a_note(marks, marks_df, capsys):
    marks.keep_rows_by_position(50, to=60)
    out = capsys.readouterr().out
    assert "The table has 54 rows, so rows 50 to 54 were kept." in out
    same(marks, marks_df.iloc[49:])


def test_positions_are_of_the_table_as_it_is_now(marks, marks_df):
    marks.keep_rows("class", equal_to="B")
    marks.keep_rows_by_position(1, to=2)
    same(marks, marks_df[marks_df["class"] == "B"].iloc[0:2])


def test_keep_rows_by_position_undo_and_steps(marks, marks_df, capsys):
    marks.keep_rows_by_position(10, to=20)
    marks.keep_rows_by_position(1, step=2)
    marks.steps()
    out = capsys.readouterr().out
    assert "1. keep rows 10 to 20" in out and "2. keep every 2nd row from row 1 to row 11" in out
    marks.undo()
    same(marks, marks_df.iloc[9:20])
    marks.undo()
    same(marks, marks_df)


def test_every_step_copies(marks):
    before = marks._frame
    marks.keep_rows_by_position(1, to=5)
    marks.replace_values("marks", above=0, with_value=0)
    assert plain(marks)["marks"].tolist() == [0] * 5
    marks.undo()
    marks.undo()
    assert marks._frame is before
    assert min(plain(marks)["marks"]) > 0


@pytest.mark.parametrize("call, message", [
    (lambda t: t.keep_rows_by_position(), "needs which rows, like 10, to=20 or [1, 5, 9]"),
    (lambda t: t.keep_rows_by_position(0), "Positions count from 1, so the first row is 1, not 0"),
    (lambda t: t.keep_rows_by_position(-1), '-1 is not a position. For the last rows use to="last" or '
                                            "table.keep_last_rows(3)"),
    (lambda t: t.keep_rows_by_position(2.5), 'Row positions are whole numbers from 1 to 54, or "last"'),
    (lambda t: t.keep_rows_by_position(True), "whole numbers from 1 to 54"),
    (lambda t: t.keep_rows_by_position("first"), "whole numbers from 1 to 54"),
    (lambda t: t.keep_rows_by_position(20, to=10), "Row 20 comes after row 10. Write the smaller first: "
                                                   "table.keep_rows_by_position(10, to=20); to turn the table "
                                                   "upside down use table.reverse_rows()"),
    (lambda t: t.keep_rows_by_position([1, 5], to=9), "to= and step= go with one starting row, not with a list"),
    (lambda t: t.keep_rows_by_position(1, step=0), "step= is how many rows to move on each time"),
    (lambda t: t.keep_rows_by_position(1, step=1.5), "step= is how many rows"),
    (lambda t: t.keep_rows_by_position([1, 5, 1]), "Row 1 is listed twice"),
    (lambda t: t.keep_rows_by_position(list(range(1, 60))), "The list has 59 positions, but there are only 54"),
    (lambda t: t.keep_rows_by_position([]), "The list of rows is empty"),
    (lambda t: t.keep_rows_by_position(60), "There is no row 60: positions count from 1, so 1 is the first row and "
                                            "54 is the last"),
    (lambda t: t.keep_rows_by_position(range(5)), "range() starts at 0"),
    (lambda t: t.keep_rows_by_position({1, 2}), "a set has no order"),
    (lambda t: t.keep_rows_by_position("marks", above=50), 'table.keep_rows("marks", above=...)'),
    (lambda t: t.keep_rows_by_position("marks"), '"marks" is a column'),
    (lambda t: t.keep_rows_by_position(1, stpe=2), "Did you mean step=?"),
    (lambda t: t.remove_rows_by_position(1, to="last"), "That would remove every row"),
    (lambda t: t.remove_rows_by_position("marks", below=50), 'table.remove_rows("marks", below=...)'),
])
def test_keep_rows_by_position_errors(marks, call, message):
    with pytest.raises(SimplydataError) as info:
        call(marks)
    assert message in str(info.value)
    assert not marks._history


def test_empty_table_has_no_positions(engine):
    t = _engine_table(pd.DataFrame({"x": pd.Series([], dtype=float)}), engine)
    with pytest.raises(SimplydataError, match="no rows to choose from"):
        t.keep_rows_by_position(1)


# ====================================================================== remove_rows_by_position
def test_remove_rows_by_position(marks, marks_df, capsys):
    marks.remove_rows_by_position([12, 40])
    assert "Removed rows 12, 40 (2 of 54 rows). 52 rows left." in capsys.readouterr().out
    same(marks, marks_df.drop(index=marks_df.index[[11, 39]]))  # pandas df.drop(index=df.index[[11, 39]])


def test_remove_the_first_row_and_every_second_row(marks, marks_df):
    marks.remove_rows_by_position(1)
    same(marks, marks_df.iloc[1:])
    marks.undo()
    marks.remove_rows_by_position(2, step=2)
    same(marks, marks_df.iloc[0::2])  # NumPy np.delete(a, range(1, 54, 2), axis=0)


def test_remove_rows_to_beyond_the_end(marks, marks_df, capsys):
    marks.remove_rows_by_position(50, to=99)
    assert "so rows 50 to 54 were removed" in capsys.readouterr().out
    same(marks, marks_df.iloc[:49])


# ====================================================================== columns by position
def test_keep_columns_by_position_names_the_columns(marks, marks_df, capsys):
    marks.keep_columns_by_position(1, to=3)
    assert "Kept columns 1 to 3: name, class, subject." in capsys.readouterr().out
    same(marks, marks_df.iloc[:, 0:3])  # Polars df.select(cs.by_index(0, 1, 2))
    marks.undo()
    marks.keep_columns_by_position("last")
    same(marks, marks_df.iloc[:, [-1]])  # Polars cs.last()
    marks.undo()
    marks.keep_columns_by_position([5, 1])
    same(marks, marks_df[["marks", "name"]])
    marks.undo()
    marks.keep_columns_by_position(1, step=2)
    same(marks, marks_df.iloc[:, ::2])


def test_remove_columns_by_position(marks, marks_df, capsys):
    marks.remove_columns_by_position(1)
    assert "Removed column 1: name. 5 columns left." in capsys.readouterr().out
    same(marks, marks_df.drop(columns=marks_df.columns[0]))
    marks.undo()
    marks.remove_columns_by_position(4, to="last")
    same(marks, marks_df.iloc[:, :3])


def test_unnamed_column_from_a_csv(engine, tmp_path):
    path = tmp_path / "with_index.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path)  # writes the index as a column with no name
    t = sdata.table(str(path), fast=engine == "polars")
    first = t.column_names()[0]
    t.remove_columns_by_position(1)
    assert t.column_names() == ["a", "b"] and first not in t.column_names()


@pytest.mark.parametrize("call, message", [
    (lambda t: t.keep_columns_by_position(0), "the first column is 1, not 0"),
    (lambda t: t.keep_columns_by_position(9), "There is no column 9"),
    (lambda t: t.keep_columns_by_position(3, to=1), "Column 3 comes after column 1"),
    (lambda t: t.keep_columns_by_position("marks"), 'table.keep_columns("marks")'),
    (lambda t: t.keep_columns_by_position(starts_with="h"), "table.keep_columns(starts_with=...)"),
    (lambda t: t.remove_columns_by_position(1, to="last"), "That would remove every column"),
])
def test_columns_by_position_errors(marks, call, message):
    with pytest.raises(SimplydataError) as info:
        call(marks)
    assert message in str(info.value)


# ====================================================================== row and cell_at
def test_row_gives_a_plain_dict(marks, marks_df):
    row = marks.row(3)
    assert row == marks_df.iloc[2].to_dict()  # pandas df.iloc[2].to_dict(); Polars df.row(2, named=True)
    assert type(row["marks"]) is int and type(row["hours"]) is float
    assert marks.row("last")["name"] == marks_df.iloc[-1]["name"]
    assert not marks._history


def test_cell_at(marks, marks_df, capsys):
    assert marks.cell_at(row=3, column="marks") == marks_df.at[2, "marks"]  # Polars df.item(2, "marks")
    assert marks.cell_at(row="last", column="name") == marks_df["name"].iloc[-1]
    assert capsys.readouterr().out == ""
    assert not marks._history


def test_cell_at_empty_cell_and_dates(engine):
    data = pd.DataFrame({"x": [1.0, np.nan], "when": pd.to_datetime(["2026-03-15", "2026-04-01"]),
                         "ok": [True, False]})
    t = _engine_table(data, engine)
    assert t.cell_at(row=2, column="x") is None
    assert t.cell_at(row=1, column="when") == dt.datetime(2026, 3, 15)
    assert t.cell_at(row=1, column="ok") is True
    assert t.row(2) == {"x": None, "when": dt.datetime(2026, 4, 1), "ok": False}


def test_cell_at_says_when_a_cell_is_empty(engine, capsys):
    t = _engine_table(pd.DataFrame({"marks": [1.0, None, 3.0]}), engine)
    t.cell_at(row=2, column="marks")
    assert 'Row 2, column "marks" is empty.' in capsys.readouterr().out


@pytest.mark.parametrize("call, message", [
    (lambda t: t.cell_at(row=3, column=2), 'column= takes the column\'s name, like column="marks"'),
    (lambda t: t.cell_at(row=3), "needs column="),
    (lambda t: t.cell_at(column="marks"), "needs row="),
    (lambda t: t.cell_at(row=99, column="marks"), "There is no row 99"),
    (lambda t: t.cell_at(row=1, column="mraks"), 'Did you mean "marks"?'),
    (lambda t: t.row(0), "the first row is 1, not 0"),
    (lambda t: t.row(), "needs which row"),
    (lambda t: t.row([1, 2]), "table.keep_rows_by_position([1, 2])"),
    (lambda t: t.row(-1), '"last" on its own is the last row'),
])
def test_row_and_cell_errors(marks, call, message):
    with pytest.raises(SimplydataError) as info:
        call(marks)
    assert message in str(info.value)


# ====================================================================== reverse_rows
def test_reverse_rows(marks, marks_df, capsys):
    marks.reverse_rows()
    assert "Reversed the order of the 54 rows: the last row is now first." in capsys.readouterr().out
    same(marks, marks_df.iloc[::-1])  # Polars df.reverse()
    marks.undo()
    same(marks, marks_df)


# ====================================================================== keep_columns: run, kind, names
def test_keep_columns_run_by_name(weather, weather_df, capsys):
    weather.keep_columns("temperature", to="wind_speed")
    assert "Kept 4 columns: temperature, rain, wind_direction, wind_speed." in capsys.readouterr().out
    same(weather, weather_df.loc[:, "temperature":"wind_speed"])  # pandas loc slice, both ends included
    weather.steps()
    assert "keep columns temperature to wind_speed" in capsys.readouterr().out


def test_keep_columns_run_of_one(weather, weather_df):
    weather.keep_columns("rain", to="rain")
    same(weather, weather_df[["rain"]])


@pytest.mark.parametrize("kind", ["number", "numbers", "numeric", "Number"])
def test_keep_columns_by_kind(weather, weather_df, kind):
    weather.keep_columns(kind=kind)
    same(weather, weather_df.select_dtypes("number"))  # Polars cs.numeric()


def test_keep_columns_text_date_true_false(engine):
    data = pd.DataFrame({"name": ["a", "b"], "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                         "ok": [True, False], "x": [1.0, 2.0]})
    for kind, expected in (("text", ["name"]), ("string", ["name"]), ("date", ["when"]), ("datetime", ["when"]),
                           ("true/false", ["ok"]), ("bool", ["ok"]), ("number", ["x"])):
        t = _engine_table(data, engine)
        t.keep_columns(kind=kind)
        assert t.column_names() == expected  # Polars cs.string() / cs.temporal() / cs.boolean() / cs.numeric()


@pytest.mark.parametrize("option, value, expected", [
    ("starts_with", "wind", ["wind_direction", "wind_speed"]),
    ("ends_with", "_speed", ["wind_speed"]),
    ("contains", "on", ["month", "month_number", "wind_direction"]),
    ("pattern", r"^wind_", ["wind_direction", "wind_speed"]),
    ("pattern", r"n$", ["rain", "wind_direction"]),
])
def test_keep_columns_by_name_text(weather, weather_df, option, value, expected):
    weather.keep_columns(**{option: value})
    assert weather.column_names() == expected
    native = {"starts_with": lambda: [c for c in weather_df.columns if c.startswith(value)],
              "ends_with": lambda: [c for c in weather_df.columns if c.endswith(value)],
              "contains": lambda: list(weather_df.filter(like=value).columns),
              "pattern": lambda: list(weather_df.filter(regex=value).columns)}[option]()
    assert expected == native


def test_two_steps_keep_columns_that_meet_both(weather):
    weather.keep_columns(kind="number")
    weather.keep_columns(starts_with="wind")
    assert weather.column_names() == ["wind_direction", "wind_speed"]


def test_remove_columns_run_kind_and_names(weather, weather_df, capsys):
    weather.remove_columns("wind_direction", to="wind_speed")
    assert "Removed 2 columns: wind_direction, wind_speed. 5 columns left." in capsys.readouterr().out
    same(weather, weather_df.drop(columns=["wind_direction", "wind_speed"]))
    weather.undo()
    weather.remove_columns(kind="text")
    same(weather, weather_df.drop(columns=["month", "city"]))  # Polars df.select(~cs.string())
    weather.undo()
    weather.remove_columns(contains="wind")
    assert weather.column_names() == ["month", "month_number", "city", "temperature", "rain"]


def test_plain_keep_columns_still_quiet_and_in_given_order(marks, capsys):
    marks.keep_columns("marks", "name")
    assert capsys.readouterr().out == ""
    assert marks.column_names() == ["marks", "name"]


@pytest.mark.parametrize("call, message", [
    (lambda t: t.keep_columns("wind_speed", to="temperature"), '"temperature" comes before "wind_speed" in the '
                                                             'table. Write the column on the left first'),
    (lambda t: t.keep_columns("rain", "city", to="wind_speed"), "to= goes with one column name before it"),
    (lambda t: t.keep_columns(to="rain"), "to= needs the first column's name before it"),
    (lambda t: t.keep_columns("rain", to=5), "to= takes the name of the last column"),
    (lambda t: t.keep_columns("rian", to="wind_speed"), 'Did you mean "rain"?'),
    (lambda t: t.keep_columns(kind="date"), "This table has no date columns. The kinds it has: text (month, city)"),
    (lambda t: t.keep_columns(kind="numbr"), 'Did you mean kind="number"?'),
    (lambda t: t.keep_columns("city", kind="text"), "one way at a time, but you gave column names and kind=. "
                                                   "Write two steps"),
    (lambda t: t.keep_columns(kind="number", starts_with="w"), "you gave kind= and starts_with="),
    (lambda t: t.keep_columns(pattern="("), "pattern= is a regular expression"),
    (lambda t: t.keep_columns(contains="zzz"), 'No column name contains "zzz"'),
    (lambda t: t.keep_columns(starts_with="Wind"), 'No column name starts with "Wind" (capital letters matter)'),
    (lambda t: t.keep_columns(starts_with=3), "starts_with= needs some text in quotes"),
    (lambda t: t.keep_columns(2), "keep_columns_by_position(2)"),
    (lambda t: t.keep_columns(), "needs the names of the columns to keep, or one of kind="),
    (lambda t: t.remove_columns(pattern="."), "That would remove every column"),
])
def test_keep_columns_errors(weather, call, message):
    with pytest.raises(SimplydataError) as info:
        call(weather)
    assert message in str(info.value)
    assert not weather._history


# ====================================================================== replace_values with conditions
def test_replace_values_above_caps(marks, marks_df, capsys):
    marks.replace_values("marks", above=90, with_value=90)
    n = int((marks_df["marks"] > 90).sum())
    assert f'Replaced {n} values in "marks" that were above 90 with 90.' in capsys.readouterr().out
    same(marks, marks_df.assign(marks=marks_df["marks"].clip(upper=90)))  # pandas clip / mask
    marks.undo()
    same(marks, marks_df)


@pytest.mark.parametrize("option, value, native", [
    ("above", 90, lambda s: s > 90), ("below", 40, lambda s: s < 40), ("at_least", 95, lambda s: s >= 95),
    ("at_most", 40, lambda s: s <= 40), ("equal_to", 61, lambda s: s == 61), ("not_equal_to", 61, lambda s: s != 61),
    ("between", [50, 60], lambda s: s.between(50, 60)), ("one_of", [61, 66], lambda s: s.isin([61, 66])),
])
def test_replace_values_each_number_condition(marks, marks_df, option, value, native):
    marks.replace_values("marks", with_value=0, **{option: value})
    expected = marks_df["marks"].mask(native(marks_df["marks"]), 0)  # pandas s.mask(cond, 0)
    assert plain(marks)["marks"].tolist() == expected.tolist()


@pytest.mark.parametrize("option, value, native", [
    ("contains", "e", lambda s: s.str.contains("e")), ("starts_with", "A", lambda s: s.str.startswith("A")),
    ("ends_with", "a", lambda s: s.str.endswith("a")),
])
def test_replace_values_text_conditions(marks, marks_df, option, value, native):
    marks.replace_values("name", with_value="X", **{option: value})
    assert plain(marks)["name"].tolist() == marks_df["name"].mask(native(marks_df["name"]), "X").tolist()


def test_replace_values_dates(engine):
    data = pd.DataFrame({"when": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-05-01"])})
    t = _engine_table(data, engine)
    t.replace_values("when", after="2026-02-01", with_value=None)
    assert plain(t)["when"].isna().tolist() == [False, True, True]
    t.undo()
    t.replace_values("when", before="2026-02-01", with_value="2026-02-01")
    assert plain(t)["when"].tolist()[0] == pd.Timestamp("2026-02-01")


def test_replace_values_with_none_makes_empties_and_empties_never_match(engine, capsys):
    data = pd.DataFrame({"rain": [5.0, -1.0, np.nan, -3.0, 2.0]})
    t = _engine_table(data, engine)
    t.replace_values("rain", below=0, with_value=None)
    assert 'Made 2 values in "rain" empty: the ones that were below 0.' in capsys.readouterr().out
    expected = data["rain"].mask(data["rain"] < 0)  # pandas s.mask(s < 0)
    assert plain(t)["rain"].isna().tolist() == expected.isna().tolist()
    t.undo()
    t.replace_values("rain", not_equal_to=5, with_value=0)
    assert plain(t)["rain"].isna().tolist() == [False, False, True, False, False]  # the empty cell stays


def test_replace_values_missing_value_codes(engine):
    t = _engine_table(pd.DataFrame({"price": [10.0, -99, 12, -999]}), engine)
    t.replace_values("price", one_of=[-99, -999], with_value=None)
    assert plain(t)["price"].isna().tolist() == [False, True, False, True]


def test_replace_values_steps_say_the_condition(marks, capsys):
    marks.replace_values("marks", above=90, with_value=90)
    marks.steps()
    assert "replace values in marks where marks is above 90 with 90" in capsys.readouterr().out


@pytest.mark.parametrize("call, message", [
    (lambda t: t.replace_values("marks", 61, above=90, with_value=0), "either the old value or one condition"),
    (lambda t: t.replace_values("marks", above=90), "needs with_value="),
    (lambda t: t.replace_values("marks", above=90, below=10, with_value=0), "one condition at a time"),
    (lambda t: t.replace_values("marks", above="x", with_value=0), "needs a number"),
    (lambda t: t.replace_values("marks", above=90, with_value="x"), "needs a number"),
    (lambda t: t.replace_values("name", above=90, with_value="x"), "compares numbers or dates"),
    (lambda t: t.replace_values("marks", abuve=90, with_value=0), "Did you mean above=?"),
])
def test_replace_values_condition_errors(marks, call, message):
    with pytest.raises(SimplydataError) as info:
        call(marks)
    assert message in str(info.value)
    assert not marks._history


def test_replace_values_old_ways_still_work(marks, marks_df):
    marks.replace_values("subject", "Maths", with_value="Math")
    assert "Math" in plain(marks)["subject"].tolist()
    marks.replace_values("class", {"A": "B", "B": "A"})
    assert plain(marks)["class"].tolist() == marks_df["class"].replace({"A": "B", "B": "A"}).tolist()


# ====================================================================== [ ] and dots
@pytest.mark.parametrize("key, first_line", [
    ("marks", 'For the column "marks" on its own write table.column("marks").'),
    ("mraks", 'There is no column "mraks". Did you mean table.column("marks")?'),
    (["name", "marks"], 'To keep these columns write table.keep_columns("name", "marks").'),
    (slice(2, 5), "In Python [2:5] means rows 3 to 5 counting from 1: table.keep_rows_by_position(3, to=5)."),
    (slice(None, 3), "rows 1 to 3 counting from 1: table.keep_rows_by_position(1, to=3)."),
    (slice(10, None), 'table.keep_rows_by_position(11, to="last")'),
    (2, "In Python [2] is row 3 counting from 1: table.row(3) shows it."),
    (-1, 'table.row("last")'),
])
def test_brackets_point_to_the_words(marks, key, first_line):
    with pytest.raises(SimplydataError) as info:
        marks[key]
    text = str(info.value)
    assert first_line in text.splitlines()[0]
    assert "Python's [ ] counts from 0, so simplydata does not use it" in text
    assert "table.keep_rows_by_position(10, to=20)" in text and 'table.cell_at(row=3, column="marks")' in text


def test_brackets_error_is_still_a_type_error(marks):
    with pytest.raises(TypeError):
        marks["marks"]
    with pytest.raises(TypeError, match="table.row"):
        list(marks)


def test_column_after_a_dot_points_to_column_and_keep_rows(marks):
    with pytest.raises(AttributeError) as info:
        marks.marks
    text = str(info.value)
    assert text.startswith('marks is a column. For its numbers write table.column("marks"); to choose rows write '
                           'table.keep_rows("marks > 50").')
    with pytest.raises(SimplydataError, match=r'For its values write table.column\("name"\)'):
        marks.name
    assert not hasattr(marks, "marks")
