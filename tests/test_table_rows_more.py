"""Tests for the v0.3 row words in rows.py: written conditions, after=/before=, biggest and
smallest rows, outliers, replace_values with {old: new}, round_to and show, on pandas and Polars."""

import datetime as dt

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
        "hours": [2.0, 3.0, 0.0, 4.0, np.nan, 7.0],
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
        g = [None if pd.isna(v) else v for v in got[c]]
        e = [None if pd.isna(v) else v for v in expected[c]]
        if expected[c].dtype.kind == "f":
            assert np.allclose(np.array(g, dtype=float), np.array(e, dtype=float), equal_nan=True), c
        else:
            assert g == e, c


# ---------------------------------------------------------------------- written conditions
WRITTEN = [
    ("marks > 50 and class == 'A'", lambda d: (d["marks"] > 50) & (d["class"] == "A")),
    ("marks > 50 or class == 'A'", lambda d: (d["marks"] > 50) | (d["class"] == "A")),
    ("not (marks > 50)", lambda d: ~(d["marks"] > 50)),
    ("(marks > 80 or marks < 40) and subject != 'English'",
     lambda d: ((d["marks"] > 80) | (d["marks"] < 40)) & (d["subject"] != "English")),
    ("subject in ['Maths', 'Science']", lambda d: d["subject"].isin(["Maths", "Science"])),
    ("subject not in ['Maths']", lambda d: ~d["subject"].isin(["Maths"])),
    ("40 <= marks <= 60", lambda d: d["marks"].between(40, 60)),
    ("50 < marks", lambda d: d["marks"] > 50),
    ("marks >= 61", lambda d: d["marks"] >= 61),
    ("`marks` == 61", lambda d: d["marks"] == 61),
    ("hours != 6.3", lambda d: d["hours"] != 6.3),
    ("marks > -1", lambda d: d["marks"] > -1),
    ("marks / hours > 10", lambda d: d["marks"] / d["hours"] > 10),
    ("hours * 10 > marks", lambda d: d["hours"] * 10 > d["marks"]),
    ("marks + height > 220", lambda d: d["marks"] + d["height"] > 220),
    ("marks > height / 2", lambda d: d["marks"] > d["height"] / 2),
    ("name == 'Asha' or name == \"Ben\"", lambda d: d["name"].isin(["Asha", "Ben"])),
    ("class == 'A' and not subject == 'Maths'", lambda d: (d["class"] == "A") & (d["subject"] != "Maths")),
    ("marks>50 and(class=='B'or class=='C')", lambda d: (d["marks"] > 50) & d["class"].isin(["B", "C"])),
]


@pytest.mark.parametrize("text, mask", WRITTEN)
def test_written_conditions_match_pandas(marks, marks_df, text, mask, capsys):
    expected = marks_df[mask(marks_df)]
    marks.keep_rows(text)
    same(marks, expected)
    out = capsys.readouterr().out
    assert f"Kept {len(expected)} of 54 rows." in out
    marks.steps()
    assert f"keep rows where {text}" in capsys.readouterr().out
    marks.undo()
    same(marks, marks_df)
    marks.remove_rows(text)
    same(marks, marks_df[~mask(marks_df)])
    marks.undo()
    same(marks, marks_df)


def test_written_conditions_and_empty_cells(messy, messy_df):
    # a comparison with an empty cell is never true, and "not" does not bring it back
    messy.keep_rows("marks > 50")
    assert len(messy) == 3
    messy.undo()
    messy.keep_rows("not (marks > 50)")
    assert list(plain(messy)["marks"]) == [45.0, 45.0]
    messy.undo()
    messy.remove_rows("marks > 50")  # the empty mark is not above 50, so it stays
    assert len(messy) == 3 and plain(messy)["marks"].isna().sum() == 1
    messy.undo()
    messy.keep_rows("marks is empty")
    assert len(messy) == 1
    messy.undo()
    messy.keep_rows("name is empty")  # None and a cell of only spaces
    assert len(messy) == 2
    messy.undo()
    messy.keep_rows("name is not empty and marks is not empty")
    assert len(messy) == 3
    messy.undo()
    # "or" still finds the row when the other part is true
    messy.keep_rows("marks > 75 or not passed")
    assert list(plain(messy)["marks"].fillna(-1)) == [-1.0, 80.0, 45.0]
    messy.undo()
    messy.keep_rows("passed")
    same(messy, messy_df[messy_df["passed"]])
    messy.undo()
    messy.keep_rows("passed == False")
    same(messy, messy_df[~messy_df["passed"]])
    messy.undo()
    # a sum with an empty cell, or a division by zero, is not known either
    messy.keep_rows("marks / hours > 0")
    assert list(plain(messy)["name"].fillna("?")) == ["Asha", "?", "  "]
    messy.undo()
    messy.keep_rows("not (marks / hours > 0)")
    assert len(messy) == 0


def test_written_conditions_on_dates(messy):
    messy.make_dates("when")
    messy.keep_rows("when > '2026-03-14'")
    assert len(messy) == 4
    messy.undo()
    messy.keep_rows("when >= '2026-03-15' and when < '2026-12-01'")
    assert len(messy) == 3
    messy.undo()
    messy.keep_rows("when == '2026-01-01' or when is empty")
    assert len(messy) == 2
    messy.undo()
    with pytest.raises(SimplydataError, match="year-month-day"):
        messy.keep_rows("when > '15 March'")
    with pytest.raises(SimplydataError, match="year-month-day"):
        messy.keep_rows("when > 2026")


def test_written_condition_with_keyword_and_spaces(engine):
    df = pd.DataFrame({"class": ["A", "B", "A"], "max temp": [10.0, 20.0, 30.0]})
    t = _engine_table(df, engine)
    t.keep_rows("class == 'A' and `max temp` > 15")
    same(t, df[(df["class"] == "A") & (df["max temp"] > 15)])
    t.undo()
    t.replace_values("class", {"B": "class B"})
    t.keep_rows("class == 'class B'")  # a column name inside quotes stays text
    assert len(t) == 1


WRITTEN_ERRORS = [
    ("marks = 50", "two equal signs"),
    ("marks > 50 && class == 'A'", "and / or as words"),
    ("marks > 50 & hours > 2", "instead of & and |"),
    ("class == 'A' AND marks > 3", "small letters"),
    ("marks <> 50", "!="),
    ("marks > 50 and (class == 'A'", "bracket is missing"),
    ("class == 'A", "quote mark is missing"),
    ("marks > '50'", "without quotes: marks > 50"),
    ("marks > 'lots'", "compare it with a number"),
    ("class == 1", "put the text in quotes"),
    ("name > 'A'", "compares numbers or dates"),
    ("marks == name", "cannot be compared"),
    ("mark > 50", 'Did you mean "marks"'),
    ("marks is 3", "write empty"),
    ("marks == empty", "is empty"),
    ("marks == None", "is empty"),
    ("marks > 50 and hours", "alone is not a condition"),
    ("3 > 2", "both values"),
    ("100 / 2 > 3", "needs a column"),
    ("name + 1 > 3", "cannot be used in a sum"),
    ("marks in 50", "in like this"),
    ("marks in [hours]", "only hold values"),
    ("marks in []", "at least one value"),
    ("marks.real > 3", "dot after a name"),
    ("__import__('os').system('echo hi') > 1", "not allowed"),
    ("(lambda: 1)() > 0", "not allowed"),
    ("open('x') == 'y'", "not allowed"),
    ("marks == true", "Write True with a capital letter"),
]


@pytest.mark.parametrize("text, message", WRITTEN_ERRORS)
def test_written_condition_errors(marks, text, message):
    with pytest.raises(SimplydataError) as error:
        marks.keep_rows(text)
    assert message in str(error.value)
    assert len(marks) == 54 and not marks._history  # errors never change the table


def test_written_condition_errors_list_what_is_allowed(marks):
    with pytest.raises(SimplydataError, match="A condition can use") as error:
        marks.keep_rows("marks = 50")
    assert "formula" not in str(error.value)
    with pytest.raises(SimplydataError, match="not both"):
        marks.keep_rows("marks > 50", above=3)
    with pytest.raises(SimplydataError, match="write the whole condition"):
        marks.keep_rows("marks")
    with pytest.raises(SimplydataError, match="very long"):
        marks.keep_rows(" or ".join(["marks > 1"] * 100))


def test_true_false_column_alone(messy, messy_df):
    messy.remove_rows("passed")
    same(messy, messy_df[~messy_df["passed"]])


# ---------------------------------------------------------------------- after= / before=
def test_after_before_on_days(messy):
    messy.make_dates("when")  # 2026-03-15 (3 times), 2026-12-25, empty, 2026-01-01
    messy.keep_rows("when", after="2026-03-15")
    assert len(messy) == 1
    messy.undo()
    messy.keep_rows("when", before="2026-03-15")
    assert len(messy) == 1
    messy.undo()
    messy.remove_rows("when", after="2026-03-14")  # the empty date stays
    assert len(messy) == 2
    messy.steps()
    messy.undo()
    messy.keep_rows("when", after=dt.date(2026, 3, 14))
    assert len(messy) == 4


def test_after_before_with_times(engine):
    df = pd.DataFrame({"when": pd.to_datetime(["2026-03-01 10:00", "2026-03-02 00:00", "2026-02-28 23:00",
                                               "2026-03-01 00:00"])})
    t = _engine_table(df, engine)
    t.keep_rows("when", after="2026-03-01")  # a whole day: after 1 March is from 2 March
    assert list(plain(t)["when"]) == [pd.Timestamp("2026-03-02")]
    t.undo()
    t.keep_rows("when", before="2026-03-01")  # before 1 March ends on 28 February
    assert list(plain(t)["when"]) == [pd.Timestamp("2026-02-28 23:00")]
    t.undo()
    t.keep_rows("when", after="2026-03-01 09:00")  # with a time: after that moment
    assert len(t) == 2
    t.undo()
    t.keep_rows("when > '2026-03-01'")  # a sign compares with midnight, like pandas
    assert len(t) == 2


def test_after_before_on_polars_dates():
    pl = pytest.importorskip("polars")
    t = make_table(pl.DataFrame({"day": [dt.date(2026, 3, d) for d in (1, 2, 3)]}))
    t.keep_rows("day", after="2026-03-01")
    assert len(t) == 2
    t.undo()
    t.keep_rows("day", before="2026-03-02")
    assert len(t) == 1
    t.undo()
    t.keep_rows("day <= '2026-03-02'")
    assert len(t) == 2


def test_after_before_errors(marks, messy):
    with pytest.raises(SimplydataError, match="for dates.*For numbers use above="):
        marks.keep_rows("marks", after=50)
    messy.make_dates("when")
    with pytest.raises(SimplydataError, match="year-month-day"):
        messy.keep_rows("when", after="March")
    with pytest.raises(SimplydataError, match="one condition at a time"):
        messy.keep_rows("when", after="2026-01-01", before="2026-12-01")


# ---------------------------------------------------------------------- biggest and smallest
def test_keep_biggest_and_smallest(marks, marks_df, capsys):
    marks.keep_biggest_rows(3, by="marks")
    expected = marks_df.sort_values("marks", ascending=False, kind="stable").head(3)
    same(marks, expected)
    assert list(plain(marks)["marks"]) == list(marks_df.nlargest(3, "marks", keep="first")["marks"])
    assert "Kept the 3 rows with the biggest marks: 3 of 54 rows." in capsys.readouterr().out
    marks.undo()
    marks.keep_smallest_rows(4, by="hours")
    same(marks, marks_df.sort_values("hours", kind="stable").head(4))
    marks.undo()
    same(marks, marks_df)


def test_keep_biggest_for_each(marks, marks_df, capsys):
    marks.keep_biggest_rows(2, by="marks", for_each="subject")
    expected = (marks_df.sort_values("marks", ascending=False, kind="stable")
                .groupby("subject", sort=False).head(2)
                .sort_values("subject", kind="stable"))
    same(marks, expected)
    out = capsys.readouterr().out
    assert "for each subject: 6 of 54 rows" in out
    marks.steps()
    assert "keep the 2 rows with the biggest marks for each subject" in capsys.readouterr().out
    marks.undo()
    marks.keep_smallest_rows(1, by="marks", for_each=["class", "subject"])
    expected = (marks_df.sort_values("marks", kind="stable")
                .groupby(["class", "subject"], sort=False).head(1)
                .sort_values(["class", "subject"], kind="stable"))
    same(marks, expected)


def test_keep_biggest_ties_and_empty(engine, capsys):
    df = pd.DataFrame({"name": list("abcdef"), "score": [5.0, 9.0, 7.0, 9.0, np.nan, 7.0]})
    t = _engine_table(df, engine)
    t.keep_biggest_rows(3, by="score")
    assert list(plain(t)["name"]) == ["b", "d", "c"]
    out = capsys.readouterr().out
    assert "Note: 1 row had the same score as the last row kept and was left out" in out
    assert "1 row with an empty score cell was left out" in out
    t.undo()
    t.keep_smallest_rows(10, by="score")  # more than there are: every row with a score
    assert list(plain(t)["name"]) == ["a", "c", "f", "b", "d"]


def test_keep_biggest_errors(marks):
    with pytest.raises(SimplydataError, match="how many rows"):
        marks.keep_biggest_rows(by="marks")
    with pytest.raises(SimplydataError, match="needs by="):
        marks.keep_biggest_rows(3)
    with pytest.raises(SimplydataError, match="whole number"):
        marks.keep_biggest_rows(2.5, by="marks")
    with pytest.raises(SimplydataError, match="text column"):
        marks.keep_smallest_rows(3, by="name")
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        marks.keep_biggest_rows(3, by="mark")
    with pytest.raises(SimplydataError, match="both the column"):
        marks.keep_biggest_rows(3, by="marks", for_each="marks")
    with pytest.raises(SimplydataError, match="no option called of="):
        marks.keep_biggest_rows(3, of="marks")
    assert len(marks) == 54 and not marks._history


# ---------------------------------------------------------------------- outliers
def _tukey(values, k=1.5):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


@pytest.fixture
def wild_df(marks_df):
    df = marks_df.copy()
    df["hours"] = df["hours"].astype(float)
    df.loc[0, "hours"] = 40.0   # far out in Maths
    df.loc[5, "hours"] = -9.0   # far out (low) in English
    df.loc[7, "hours"] = np.nan  # empty: never an outlier
    return df


def test_outliers_tukey_whole_column(engine, wild_df, capsys):
    t = _engine_table(wild_df, engine)
    low, high = _tukey(wild_df["hours"])
    mask = (wild_df["hours"] < low) | (wild_df["hours"] > high)
    assert t.show_rows_with_outliers("hours") is None
    out = capsys.readouterr().out
    assert "Tukey's rule" in out and "IQR" in out and "Q3 - Q1" in out
    assert f"below {round(low, 2)} or above {round(high, 2)}" in out
    assert f"Found {int(mask.sum())} rows with an outlier, of 54 rows." in out
    assert len(t) == 54 and not t._history  # show_ never changes the table
    t.remove_rows_with_outliers("hours")
    same(t, wild_df[~mask])
    assert f"Removed {int(mask.sum())} of 54 rows." in capsys.readouterr().out
    t.steps()
    assert "remove rows with outliers in hours (1.5 x IQR)" in capsys.readouterr().out
    t.undo()
    same(t, wild_df)


def test_outliers_for_each_group(engine, wild_df, capsys):
    t = _engine_table(wild_df, engine)
    keep = pd.Series(True, index=wild_df.index)
    for _, part in wild_df.groupby("subject"):
        low, high = _tukey(part["hours"])
        keep[part.index] = ~((part["hours"] < low) | (part["hours"] > high))
    t.remove_rows_with_outliers("hours", for_each="subject")
    same(t, wild_df[keep])
    out = capsys.readouterr().out
    assert "For each subject:" in out and "Maths: below" in out
    assert (~keep).sum() >= 2


def test_outliers_three_iqr_and_sd(engine, capsys):
    rng = np.random.default_rng(3)
    values = np.append(rng.normal(50, 5, 200), [95.0, 20.0, 66.0])
    df = pd.DataFrame({"x": values})
    t = _engine_table(df, engine)
    low, high = _tukey(values, k=3)
    t.remove_rows_with_outliers("x", rule="3 IQR")
    same(t, df[(df["x"] >= low) & (df["x"] <= high)])
    t.undo()
    mean, sd = values.mean(), values.std(ddof=1)
    t.remove_rows_with_outliers("x", rule="3 standard deviations")
    same(t, df[(df["x"] >= mean - 3 * sd) & (df["x"] <= mean + 3 * sd)])
    out = capsys.readouterr().out
    assert "3 standard deviations" in out and f"average = {round(mean, 2)}" in out
    assert "Note: with" not in out  # 203 numbers are plenty
    t.undo()
    t.remove_rows_with_outliers("x", rule="2.5 x sd")
    same(t, df[(df["x"] >= mean - 2.5 * sd) & (df["x"] <= mean + 2.5 * sd)])


def test_outliers_sd_rule_warns_for_small_groups(marks, capsys):
    marks.show_rows_with_outliers("marks", for_each="subject", rule="3 standard deviations")
    # 18 marks per subject: (18 - 1) / sqrt(18) = 4.0 > 3, so no note
    assert "Note: with" not in capsys.readouterr().out
    marks.show_rows_with_outliers("marks", for_each=["class", "subject"], rule="3 standard deviations")
    # 6 marks per group: (6 - 1) / sqrt(6) = 2.04, so 3 standard deviations can never be reached
    assert "Note: with 6 numbers in a group no value can be more than 3" in capsys.readouterr().out


def test_outlier_errors(marks):
    with pytest.raises(SimplydataError, match="among numbers"):
        marks.show_rows_with_outliers("name")
    with pytest.raises(SimplydataError, match='rule="1.5 IQR"'):
        marks.remove_rows_with_outliers("marks", rule="far away")
    with pytest.raises(SimplydataError, match="does not know"):
        marks.remove_rows_with_outliers("marks", rule="0 IQR")
    with pytest.raises(SimplydataError, match="both the column"):
        marks.remove_rows_with_outliers("marks", for_each="marks")
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        marks.remove_rows_with_outliers("mark")
    with pytest.raises(SimplydataError, match="column name in quotes"):
        marks.remove_rows_with_outliers()
    assert len(marks) == 54 and not marks._history


# ---------------------------------------------------------------------- replace_values with {old: new}
def test_replace_values_with_pairs(marks, marks_df, capsys):
    marks.replace_values("class", {"A": "B", "B": "A"})  # a swap: every cell looked up once
    expected = marks_df.copy()
    expected["class"] = marks_df["class"].replace({"A": "B", "B": "A"})
    same(marks, expected)
    out = capsys.readouterr().out
    assert 'Replaced 36 cells in class: "A" became "B" (18 cells), "B" became "A" (18 cells).' in out
    marks.undo()
    marks.replace_values("subject", {"Maths": "Math", "Art": "Arts", "English": None})
    expected = marks_df.copy()
    expected["subject"] = marks_df["subject"].replace({"Maths": "Math"}).where(marks_df["subject"] != "English")
    same(marks, expected)
    out = capsys.readouterr().out
    assert 'Note: "Art" was not found in subject.' in out
    marks.steps()
    assert 'replace "Maths" with "Math", "Art" with "Arts", "English" with empty in subject' in capsys.readouterr().out
    marks.undo()
    marks.replace_values("marks", {61: 60, 66: 65})
    expected = marks_df.copy()
    expected["marks"] = marks_df["marks"].replace({61: 60, 66: 65})
    same(marks, expected)


def test_replace_values_pair_errors(marks):
    with pytest.raises(SimplydataError, match="leave out with_value="):
        marks.replace_values("class", {"A": "B"}, with_value="C")
    with pytest.raises(SimplydataError, match="brackets are empty"):
        marks.replace_values("class", {})
    with pytest.raises(SimplydataError, match="same kind"):
        marks.replace_values("class", {"A": 1})
    with pytest.raises(SimplydataError, match="needs text in quotes"):
        marks.replace_values("class", {1: "A"})
    with pytest.raises(SimplydataError, match="old: new"):
        marks.replace_values("class", "A")
    assert len(marks) == 54 and not marks._history


# ---------------------------------------------------------------------- round_to and show
def test_round_to(marks, marks_df, capsys):
    marks.round_to("hours", 0)
    expected = marks_df.copy()
    expected["hours"] = np.round(marks_df["hours"].to_numpy(), 0)
    same(marks, expected)
    assert "Rounded hours to 0 decimal places." in capsys.readouterr().out
    marks.undo()
    marks.round_to(-1)  # every number column, to the nearest 10
    expected = marks_df.copy()
    for c in ("hours", "marks", "height"):
        expected[c] = np.round(marks_df[c].to_numpy(), -1)
    same(marks, expected)
    assert plain(marks)["marks"].dtype.kind == "i"  # whole numbers stay whole numbers
    assert "to the nearest 10" in capsys.readouterr().out
    marks.undo()
    marks.round_to(["hours"], decimals=1)
    same(marks, marks_df)  # already one decimal place
    marks.undo()
    same(marks, marks_df)


def test_round_to_halves_go_to_even(engine):
    df = pd.DataFrame({"x": [0.5, 1.5, 2.5, -2.5, 3.5, 0.125, 2.675]})
    t = _engine_table(df, engine)
    t.round_to("x")
    assert list(plain(t)["x"]) == list(np.round(df["x"].to_numpy(), 0)) == [0, 2, 2, -2, 4, 0, 3]
    t.undo()
    t.round_to(2)
    assert np.allclose(plain(t)["x"], np.round(df["x"].to_numpy(), 2))


def test_round_to_errors(marks):
    with pytest.raises(SimplydataError, match="rounds numbers"):
        marks.round_to("name", 1)
    with pytest.raises(SimplydataError, match="whole number of decimal places"):
        marks.round_to("hours", 1.5)
    with pytest.raises(SimplydataError, match='Did you mean "hours"'):
        marks.round_to("hour", 1)
    with pytest.raises(SimplydataError, match="no columns of numbers"):
        make_table(pd.DataFrame({"a": ["x"]})).round_to(1)
    assert not marks._history


def test_show(marks, marks_df, capsys):
    assert marks.show() is None
    out = capsys.readouterr().out
    assert out.count("Asha") == (marks_df["name"] == "Asha").sum()
    assert "subject" in out and "height" in out
    marks.show(rows=3)
    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if line.strip()]) == 5  # header, 3 rows, note
    assert "the first 3 rows of 54" in out
    with pytest.raises(SimplydataError, match="whole number of rows"):
        marks.show(rows="all")
    assert not marks._history


def test_show_long_table(engine, capsys):
    t = _engine_table(pd.DataFrame({"x": range(1500)}), engine)
    t.show()
    out = capsys.readouterr().out
    assert " 999" in out and "1000 " not in out.splitlines()[-2]
    assert "the first 1000 rows of 1500" in out


def test_summary_words_are_refused_for_now(marks):
    with pytest.raises(SimplydataError, match="add_column"):
        marks.keep_rows("marks > average(marks)")
    assert not marks._history


# ---------------------------------------------------------------------- line between neighbours
def test_fill_with_the_line_between_neighbours_matches_pandas(engine, capsys):
    data = pd.DataFrame({"v": [np.nan, 2.0, np.nan, np.nan, 8.0, np.nan], "w": [1, 2, 3, 4, 5, 6]})
    t = _engine_table(data, engine)
    t.fill_empty_cells("v", with_the="line between neighbours")
    expected = data["v"].interpolate(method="linear", limit_area="inside")
    assert np.allclose(plain(t)["v"], expected, equal_nan=True)
    out = capsys.readouterr().out
    assert "Filled 2 cells in v on the line between neighbours in table order." in out
    assert "Note: 2 cells stayed empty: they are at the start or end" in out
    t.steps()
    assert "with the line between neighbours" in capsys.readouterr().out


def test_fill_with_the_line_follows_dates_and_groups(engine, capsys):
    # shop B's rows are mixed in, and the days are not in order; 4 and 5 March are missing
    data = pd.DataFrame({
        "shop": ["A", "B", "A", "A", "B", "A", "B"],
        "date": pd.to_datetime(["2026-03-06", "2026-03-01", "2026-03-01", "2026-03-02", "2026-03-03",
                                "2026-03-03", "2026-03-02"]),
        "sales": [60.0, 5.0, 10.0, np.nan, 7.0, np.nan, np.nan],
    })
    t = _engine_table(data, engine)
    t.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date", for_each="shop")
    # pandas: per shop, by date, spaced by the dates (method="time")
    expected = (data.set_index("date").groupby("shop")["sales"]
                .transform(lambda s: s.sort_index().interpolate(method="time", limit_area="inside")
                           .reindex(s.index)))
    got = plain(t)["sales"]
    assert np.allclose(got, expected.to_numpy(), equal_nan=True)
    assert got[3] == pytest.approx(20.0) and got[5] == pytest.approx(30.0) and got[6] == pytest.approx(6.0)
    assert list(plain(t)["shop"]) == list(data["shop"])  # rows keep their place
    assert "in order of date, for each shop on its own" in capsys.readouterr().out


def test_fill_with_the_line_errors(marks):
    with pytest.raises(SimplydataError, match="needs numbers"):
        marks.fill_empty_cells("name", with_the="line between neighbours")
    with pytest.raises(SimplydataError, match="numbers or dates"):
        marks.fill_empty_cells("marks", with_the="line between neighbours", sorted_by="name")
    with pytest.raises(SimplydataError, match='go with with_the="line between neighbours"'):
        marks.fill_empty_cells("marks", with_the="average", for_each="class")
    with pytest.raises(SimplydataError, match="line between neighbours"):
        marks.fill_empty_cells("marks", with_the="nearest")
