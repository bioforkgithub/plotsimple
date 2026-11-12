"""Tests for the newer column words: date parts, days between dates, text words, change_column,
group rules, by_running=, summary words with for_each=, and rolling / previous / change columns.

Every word runs on pandas and Polars and is checked against plain pandas / NumPy.
"""

import re

import numpy as np
import pandas as pd
import pytest

from simplydata import SimplydataError
from simplydata.table import table as make_table

ENGINES = ["pandas", "polars"]


def build(data, engine, dates_only=()):
    data = pd.DataFrame(data)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        native = pl.from_pandas(data)
        if dates_only:
            native = native.with_columns([pl.col(c).dt.date() for c in dates_only])
        return make_table(native)
    return make_table(data)


def plain(t):
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
            assert x == y, (x, y)
        else:
            assert x == pytest.approx(y), (x, y)


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


@pytest.fixture
def table(engine, marks):
    return build(marks, engine)


def undo_gives_back(t, before):
    t.undo()
    pd.testing.assert_frame_equal(plain(t), before, check_dtype=False)


# ---------------------------------------------------------------------------- dates
STAMPS = pd.to_datetime(
    ["2026-01-01 08:30", "2026-03-15 23:05", None, "2024-12-30 00:00", "2027-01-01 12:00", "2026-06-07 17:45"],
    format="ISO8601",
)


def test_date_parts_match_pandas(engine):
    t = build({"when": STAMPS}, engine)
    s = pd.Series(STAMPS)
    expected = {
        "year": s.dt.year, "quarter": s.dt.quarter, "month": s.dt.month, "month name": s.dt.month_name(),
        "week": s.dt.isocalendar().week.astype("float64"), "day": s.dt.day,
        "weekday": s.dt.day_of_week + 1, "weekday name": s.dt.day_name(), "day of year": s.dt.day_of_year,
        "hour": s.dt.hour, "minute": s.dt.minute,
    }
    for part, values in expected.items():
        t.add_date_part_column(part, from_column="when", part=part)
        same_values(plain(t)[part], values)
    # ISO week: 30 Dec 2024 is in week 1 of 2025, 1 Jan 2027 is in week 53 of 2026
    assert plain(t)["week"][3] == 1 and plain(t)["week"][4] == 53


def test_date_parts_whole_numbers_without_empty_dates(engine):
    t = build({"when": pd.to_datetime(["2026-01-05", "2026-02-10"])}, engine)
    t.add_date_part_column("year", from_column="when", part="year")
    t.add_date_part_column("week", from_column="when", part="week")
    out = plain(t)
    assert list(out["year"]) == [2026, 2026] and list(out["week"]) == [2, 7]
    assert pd.api.types.is_integer_dtype(out["year"]) and pd.api.types.is_integer_dtype(out["week"])


def test_date_part_errors_and_undo(engine):
    t = build({"when": STAMPS, "text": ["2026-01-01"] * 6}, engine)
    before = plain(t)
    with pytest.raises(SimplydataError, match="make_dates"):
        t.add_date_part_column("y", from_column="text", part="year")
    with pytest.raises(SimplydataError, match='Did you mean part="month"'):
        t.add_date_part_column("m", from_column="when", part="mont")
    with pytest.raises(SimplydataError, match="needs part="):
        t.add_date_part_column("m", from_column="when")
    with pytest.raises(SimplydataError, match="from_column="):
        t.add_date_part_column("m", part="month")
    with pytest.raises(SimplydataError, match="Did you mean"):
        t.add_date_part_column("m", from_column="whn", part="month")
    t.add_date_part_column("m", from_column="when", part="month")
    assert "add date part column" in t._history[-1][0]
    undo_gives_back(t, before)


def test_hour_of_plain_dates_is_refused():
    pytest.importorskip("polars")
    t = build({"day": pd.to_datetime(["2026-01-01"])}, "polars", dates_only=["day"])
    with pytest.raises(SimplydataError, match="no hours"):
        t.add_date_part_column("h", from_column="day", part="hour")
    t.add_date_part_column("d", from_column="day", part="weekday name")
    assert plain(t)["d"][0] == "Thursday"


def test_days_between_two_dates(engine, capsys):
    start = pd.to_datetime(["2026-01-01", "2026-03-01 12:00", "2026-05-01", None], format="ISO8601")
    end = pd.to_datetime(["2026-01-31", "2026-03-03 06:00", "2026-04-29 18:00", "2026-01-01"], format="ISO8601")
    t = build({"start": start, "end": end}, engine)
    before = plain(t)
    t.add_column("days", by_subtracting=["end", "start"])
    assert "whole days from start to end" in capsys.readouterr().out
    expected = np.trunc((pd.Series(end) - pd.Series(start)) / pd.Timedelta(days=1))
    same_values(plain(t)["days"], expected)  # 30, 1 (1.75 days), -1 (-1.25 days), empty
    assert t._history[-1][0] == 'add column "days" = end - start (whole days)'
    undo_gives_back(t, before)


def test_days_between_plain_dates_is_whole_numbers():
    pytest.importorskip("polars")
    data = {"start": pd.to_datetime(["2026-01-01", "2026-02-01"]), "end": pd.to_datetime(["2026-03-01", "2026-02-01"])}
    t = build(data, "polars", dates_only=["start", "end"])
    t.add_column("days", by_subtracting=["end", "start"])
    assert list(plain(t)["days"]) == [59, 0]


def test_days_between_errors(engine):
    t = build({"start": STAMPS, "end": STAMPS, "n": range(6)}, engine)
    with pytest.raises(SimplydataError, match="n is not dates"):
        t.add_column("x", by_subtracting=["end", "n"])
    with pytest.raises(SimplydataError, match="two date columns"):
        t.add_column("x", by_adding=["end", "start"])
    with pytest.raises(SimplydataError, match="two date columns"):
        t.add_column("x", by_subtracting=["end", "start", "end"])


# ---------------------------------------------------------------------------- joining and splitting text
def test_join_text(engine):
    t = build({"first": ["Asha", "Ben", None, " ", None], "last": ["Rao", None, "Lee", "Kim", None], "n": [1, 2, 3, 4, 5]},
              engine)
    t.add_column("full", by_joining_text=["first", "last"], separator=" ")
    same_values(plain(t)["full"], ["Asha Rao", "Ben", "Lee", "Kim", np.nan])
    t.add_column("code", by_joining_text=["last", "n"], separator="-")
    same_values(plain(t)["code"], ["Rao-1", "2", "Lee-3", "Kim-4", "5"])
    t.add_column("plain", by_joining_text=["first", "last"])  # a space by default
    assert plain(t)["plain"][0] == "Asha Rao"
    assert t._history[-1][0] == 'add column "plain" = first, last joined with " "'


def test_join_text_on_marks(table, marks):
    table.add_column("name_class", by_joining_text=["name", "class"], separator=", ")
    same_values(plain(table)["name_class"], marks["name"] + ", " + marks["class"])
    with pytest.raises(SimplydataError, match="separator= only works together with by_joining_text="):
        table.add_column("x", by_adding=["marks", "test"], separator=" ")
    with pytest.raises(SimplydataError, match="two or more column names"):
        table.add_column("x", by_joining_text=["name"])


SAMPLES = ["Smith, John", "Lee,Ann, Jr", "Kim", None, "", " Rao , Asha "]


def test_split_column_matches_pandas(engine, capsys):
    t = build({"id": range(6), "who": SAMPLES, "after": range(6)}, engine)
    before = plain(t)
    t.split_column("who", separator=",", into=["last", "first"])
    out = plain(t)
    assert list(out.columns) == ["id", "last", "first", "after"]
    expected = pd.Series(SAMPLES, dtype=object).str.split(",", n=1, expand=True)
    for i, name in enumerate(["last", "first"]):
        exp = expected[i].str.strip().replace("", np.nan)
        same_values(out[name], exp)
    printed = capsys.readouterr().out
    assert 'Split "who" at "," into last, first.' in printed
    assert "1 cell had fewer than 2 pieces" in printed  # "Kim" ("" counts as empty)
    assert "1 cell had more than 2 pieces; the rest stays in first" in printed
    assert out["first"][1] == "Ann, Jr"
    undo_gives_back(t, before)


def test_split_column_can_keep_the_name_and_errors(engine):
    t = build({"name": ["a_b", "c_d"], "n": [1, 2]}, engine)
    with pytest.raises(SimplydataError, match="already has a column called n"):
        t.split_column("name", separator="_", into=["n", "second"])
    with pytest.raises(SimplydataError, match="two or more names"):
        t.split_column("name", separator="_", into=["one"])
    with pytest.raises(SimplydataError, match="works on text"):
        t.split_column("n", separator="_", into=["a", "b"])
    with pytest.raises(SimplydataError, match="separator="):
        t.split_column("name", separator="", into=["a", "b"])
    with pytest.raises(SimplydataError, match="different names"):
        t.split_column("name", separator="_", into=["a", "a"])
    t.split_column("name", separator="_", into=["name", "second"])
    assert list(plain(t).columns) == ["name", "second", "n"]
    assert list(plain(t)["name"]) == ["a", "c"]


# ---------------------------------------------------------------------------- replace_text
WORDS = ["Gadus sp.", "Sprattus sp. x", None, "sp.sp.", "Clupea", ""]


def test_replace_text_plain(engine, capsys):
    t = build({"species": WORDS}, engine)
    before = plain(t)
    t.replace_text("species", " sp.", with_value="")
    same_values(plain(t)["species"], pd.Series(WORDS, dtype=object).str.replace(" sp.", "", regex=False))
    assert 'Replaced " sp." in 2 cells of species.' in capsys.readouterr().out
    undo_gives_back(t, before)
    t.replace_text("species", ".", with_value="$1\\")  # plain text: no pattern signs
    assert plain(t)["species"][0] == "Gadus sp$1\\"


def test_replace_text_pattern(engine):
    t = build({"id": ["a  1", "b 22", None, "c3"]}, engine)
    t.replace_text("id", pattern=r"\s+", with_value="_")
    same_values(plain(t)["id"], ["a_1", "b_22", np.nan, "c3"])
    t.replace_text("id", pattern=r"(\d)", with_value="$1\\x")  # the new text is used as written
    same_values(plain(t)["id"], [re.sub(r"(\d)", lambda m: "$1\\x", s) for s in ["a_1", "b_22"]] + [np.nan, "c$1\\x"])


def test_replace_text_errors(engine):
    t = build({"species": WORDS, "n": range(6)}, engine)
    with pytest.raises(SimplydataError, match="with_value="):
        t.replace_text("species", "sp.")
    with pytest.raises(SimplydataError, match="one of the two"):
        t.replace_text("species", "sp.", with_value="", pattern="x")
    with pytest.raises(SimplydataError, match="works on text"):
        t.replace_text("n", "1", with_value="2")
    with pytest.raises(SimplydataError, match="not a regular expression"):
        t.replace_text("species", pattern="(", with_value="")
    with pytest.raises(SimplydataError, match="no option called with_text"):
        t.replace_text("species", "sp.", with_text="")


def test_replace_text_pattern_the_fast_engine_cannot_use():
    pytest.importorskip("polars")
    t = build({"s": ["a1"]}, "polars")
    with pytest.raises(SimplydataError, match="look-ahead"):
        t.replace_text("s", pattern=r"(?=1)", with_value="x")


# ---------------------------------------------------------------------------- add_text_part_column
CODES = ["P12_north_3", "P7_south", "x", None, "", "_lead"]


def test_text_parts_match_pandas(engine, capsys):
    t = build({"code": CODES}, engine)
    s = pd.Series(CODES, dtype=object)

    def clean(series):
        return series.where(series.str.strip() != "", np.nan)

    t.add_text_part_column("first3", from_column="code", first_characters=3)
    same_values(plain(t)["first3"], clean(s.str[:3]))
    t.add_text_part_column("last2", from_column="code", last_characters=2)
    same_values(plain(t)["last2"], clean(s.str[-2:]))
    t.add_text_part_column("mid", from_column="code", characters=[2, 4])
    same_values(plain(t)["mid"], clean(s.str[1:4]))
    t.add_text_part_column("before", from_column="code", before="_")
    found = s.str.contains("_", regex=False) == True  # noqa: E712
    same_values(plain(t)["before"], clean(s.str.partition("_")[0].where(found)))
    t.add_text_part_column("after", from_column="code", after="_")
    same_values(plain(t)["after"], clean(s.str.partition("_")[2].where(found)))
    capsys.readouterr()
    t.add_text_part_column("plate", from_column="code", pattern=r"P\d+")
    same_values(plain(t)["plate"], s.str.extract(r"(P\d+)", expand=False))
    assert '4 cells of code had no match for the pattern, so "plate" is empty there.' not in capsys.readouterr().out
    t.add_text_part_column("number", from_column="code", pattern=r"P(\d+)")
    same_values(plain(t)["number"], s.str.extract(r"P(\d+)", expand=False))


def test_text_part_feedback_and_marks(table, marks, capsys):
    table.add_text_part_column("initial", from_column="name", first_characters=1)
    same_values(plain(table)["initial"], marks["name"].str[0])
    table.add_text_part_column("x", from_column="name", after="s")
    assert 'cells of name had no "s", so "x" is empty there.' in capsys.readouterr().out
    same_values(plain(table)["x"], marks["name"].map(lambda v: v.split("s", 1)[1] if "s" in v else np.nan))


def test_text_part_errors(engine):
    t = build({"code": CODES, "n": range(6)}, engine)
    with pytest.raises(SimplydataError, match="Choose one"):
        t.add_text_part_column("x", from_column="code")
    with pytest.raises(SimplydataError, match="2 ways at once"):
        t.add_text_part_column("x", from_column="code", before="_", after="_")
    with pytest.raises(SimplydataError, match="whole number"):
        t.add_text_part_column("x", from_column="code", first_characters=0)
    with pytest.raises(SimplydataError, match="counting from 1"):
        t.add_text_part_column("x", from_column="code", characters=[3, 1])
    with pytest.raises(SimplydataError, match="works on text"):
        t.add_text_part_column("x", from_column="n", first_characters=1)
    with pytest.raises(SimplydataError, match="the mark in quotes"):
        t.add_text_part_column("x", from_column="code", before="")


def test_text_part_marks_with_pattern_signs(engine):
    t = build({"v": ["a.b(c", "x(y"]}, engine)
    t.add_text_part_column("left", from_column="v", before="(")
    t.add_text_part_column("right", from_column="v", after=".")
    same_values(plain(t)["left"], ["a.b", "x"])
    same_values(plain(t)["right"], ["b(c", np.nan])


# ---------------------------------------------------------------------------- change_column
def test_change_column(table, marks):
    before = plain(table)
    table.change_column("marks", "marks / 10")
    out = plain(table)
    assert list(out.columns) == list(marks.columns)
    same_values(out["marks"], marks["marks"] / 10)
    assert table._history[-1][0] == 'change column "marks" = marks / 10'
    undo_gives_back(table, before)
    table.change_column("marks", from_column="marks", plus=5)
    same_values(plain(table)["marks"], marks["marks"] + 5)
    table.change_column("name", by_joining_text=["name", "class"], separator="/")
    same_values(plain(table)["name"], marks["name"] + "/" + marks["class"])
    table.change_column("height", value=0)
    assert set(plain(table)["height"]) == {0}


def test_change_column_centred_per_group(table, marks):
    table.change_column("marks", "marks - average(marks)", for_each="class")
    same_values(plain(table)["marks"], marks["marks"] - marks.groupby("class")["marks"].transform("mean"))


def test_change_column_errors(table):
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        table.change_column("mark", "marks * 2")
    with pytest.raises(SimplydataError, match="needs the name of the column"):
        table.change_column(formula="marks * 2")
    with pytest.raises(SimplydataError, match="Choose one way"):
        table.change_column("marks")


# ---------------------------------------------------------------------------- summary words and for_each
def test_share_of_group_total(table, marks):
    table.add_column("share", "marks / total(marks) * 100", for_each="class")
    expected = marks["marks"] / marks.groupby("class")["marks"].transform("sum") * 100
    same_values(plain(table)["share"], expected)
    assert table._history[-1][0] == 'add column "share" = marks / total(marks) * 100 (summaries for each class)'
    assert not any(c.startswith("_simplydata") for c in plain(table).columns)


@pytest.mark.parametrize("word,how", [("average", "mean"), ("median", "median"), ("total", "sum"),
                                      ("smallest", "min"), ("biggest", "max"), ("count", "count")])
def test_every_summary_word(engine, word, how):
    data = pd.DataFrame({"g": ["a", "a", "b", "b", "b", None], "x": [1.0, 4.0, np.nan, 2.0, 7.0, 3.0]})
    t = build(data, engine)
    t.add_column("s", f"{word}(x)", for_each="g")
    expected = data.groupby("g", dropna=False)["x"].transform(how)
    same_values(plain(t)["s"], expected)
    t.add_column("whole", f"{word}(x)")
    same_values(plain(t)["whole"], [getattr(data["x"], how)()] * len(data))


def test_summary_words_more(table, marks):
    table.add_column("from_mean", "marks - average(marks)")
    same_values(plain(table)["from_mean"], marks["marks"] - marks["marks"].mean())
    table.add_column("n", "count(name)", for_each=["class", "test"])  # count may count text
    same_values(plain(table)["n"], marks.groupby(["class", "test"])["name"].transform("count"))
    table.add_column("z", "(marks - average(marks)) / (biggest(marks) - smallest(marks))", for_each="class")
    g = marks.groupby("class")["marks"]
    same_values(plain(table)["z"], (marks["marks"] - g.transform("mean")) / (g.transform("max") - g.transform("min")))
    table.add_column("spread", "total(marks * 2)", for_each="class")
    same_values(plain(table)["spread"], g.transform("sum") * 2)


def test_summary_word_errors(table):
    with pytest.raises(SimplydataError, match="only matters when the formula has a summary word"):
        table.add_column("x", "marks * 2", for_each="class")
    with pytest.raises(SimplydataError, match="only works with a formula"):
        table.add_column("x", by_adding=["marks", "test"], for_each="class")
    with pytest.raises(SimplydataError, match="called average"):
        table.add_column("x", "marks / mean(marks)")
    with pytest.raises(SimplydataError, match="cannot be inside another summary word"):
        table.add_column("x", "total(average(marks))")
    with pytest.raises(SimplydataError, match="takes one column"):
        table.add_column("x", "total(marks, test)")
    with pytest.raises(SimplydataError, match="cannot do sums"):
        table.add_column("x", "total(name)")
    with pytest.raises(SimplydataError, match='Did you mean "class"'):
        table.add_column("x", "total(marks)", for_each="clas")
    with pytest.raises(SimplydataError, match="needs brackets"):
        table.add_column("x", "marks / total")


# ---------------------------------------------------------------------------- by_running
def test_by_running_matches_apply(table, marks):
    def per_hour(row):
        return row["marks"] / row["test"]

    before = plain(table)
    table.add_column("per_test", by_running=per_hour)
    same_values(plain(table)["per_test"], marks.apply(per_hour, axis=1))
    assert table._history[-1][0] == 'add column "per_test" = per_hour(row) for each row'
    undo_gives_back(table, before)
    table.add_column("label", by_running=lambda row: "high" if row["marks"] >= 60 else "low")
    same_values(plain(table)["label"], np.where(marks["marks"] >= 60, "high", "low"))
    table.change_column("test", by_running=lambda row: row["test"] * 10)
    same_values(plain(table)["test"], marks["test"] * 10)


def test_by_running_empty_cells_and_kinds(engine):
    t = build({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"], "n": [1, 2, 3]}, engine)
    seen = []

    def look(row):
        seen.append(row)
        return None if row["a"] is None else row["a"] * 2

    t.add_column("double", by_running=look)
    same_values(plain(t)["double"], [2.0, np.nan, 6.0])
    assert seen[1]["a"] is None and seen[1]["b"] is None and seen[0]["n"] == 1
    t.add_column("mixed_numbers", by_running=lambda row: 1 if row["n"] == 1 else 2.5)
    same_values(plain(t)["mixed_numbers"], [1.0, 2.5, 2.5])
    t.add_column("flag", by_running=lambda row: row["n"] > 1)
    assert list(plain(t)["flag"]) == [False, True, True]


def test_by_running_on_a_pandas_table_with_its_own_row_numbers():
    data = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    t = make_table(data)
    t.add_column("y", by_running=lambda row: row["x"] + 1)
    assert list(plain(t)["y"]) == [2, 3, 4]


def test_by_running_errors(table):
    def broken(row):
        return row["marks"] / 0

    with pytest.raises(SimplydataError, match=r"(?s)broken\(\) stopped at row 1 .*ZeroDivisionError"):
        table.add_column("x", by_running=broken)
    with pytest.raises(SimplydataError, match="without quotes and without brackets"):
        table.add_column("x", by_running="broken")
    with pytest.raises(SimplydataError, match="mix of kinds"):
        table.add_column("x", by_running=lambda row: "a" if row["marks"] > 60 else 1)
    with pytest.raises(SimplydataError, match="gave back a list"):
        table.add_column("x", by_running=lambda row: [1])


# ---------------------------------------------------------------------------- add_group_column rules
def test_rules_match_np_select(table, marks, capsys):
    before = plain(table)
    table.add_group_column("level", rules={"marks >= 80": "top", "marks >= 50": "pass"}, otherwise="fail")
    expected = np.select([marks["marks"] >= 80, marks["marks"] >= 50], ["top", "pass"], default="fail")
    same_values(plain(table)["level"], expected)
    printed = capsys.readouterr().out
    assert f'"top": {int((expected == "top").sum())} rows' in printed
    assert "add group column \"level\" by rules (marks >= 80 -> \"top\"" in table._history[-1][0]
    undo_gives_back(table, before)


def test_rules_list_text_and_or_not(table, marks):
    table.add_group_column("g", rules=[('class == "A" and marks >= 60', 1), ("not (marks < 40) or test == 3", 2)])
    m = marks
    expected = np.select([(m["class"] == "A") & (m["marks"] >= 60), ~(m["marks"] < 40) | (m["test"] == 3)],
                         [1, 2], default=np.nan)
    same_values(plain(table)["g"], expected)
    table.add_group_column("band", rules={"40 <= marks < 60": "middle", "marks / test > 40": "fast"})
    expected = np.select([(m["marks"] >= 40) & (m["marks"] < 60), m["marks"] / m["test"] > 40],
                         ["middle", "fast"], default=None)
    same_values(plain(table)["band"], pd.Series(expected).replace({None: np.nan}))
    table.add_group_column("is_a", rules={"class != 'A'": "no", "'A' == class": "yes"})
    same_values(plain(table)["is_a"], np.where(m["class"] == "A", "yes", "no"))


def test_rules_with_empty_cells_and_dates(engine, capsys):
    data = {"x": [10.0, np.nan, 90.0], "day": pd.to_datetime(["2026-01-01", "2026-05-01", None])}
    t = build(data, engine)
    t.add_group_column("size", rules={"x >= 50": "big"}, otherwise="small")
    same_values(plain(t)["size"], ["small", "small", "big"])
    assert "1 row have an empty cell in x" in capsys.readouterr().out
    t.add_group_column("half", rules={'day < "2026-03-01"': "first"}, otherwise="second")
    same_values(plain(t)["half"], ["first", "second", "second"])
    t.add_group_column("only", rules={"x >= 50": "big"})
    same_values(plain(t)["only"], [np.nan, np.nan, "big"])
    assert "no group (empty): 2 rows" in capsys.readouterr().out


def test_rules_errors(table):
    with pytest.raises(SimplydataError, match="two equal signs"):
        table.add_group_column("g", rules={"class = 'A'": "a"})
    with pytest.raises(SimplydataError, match="and / or instead of"):
        table.add_group_column("g", rules={"marks > 5 & test > 1": "a"})
    with pytest.raises(SimplydataError, match="compare it with text in quotes"):
        table.add_group_column("g", rules={"class == 5": "a"})
    with pytest.raises(SimplydataError, match="without quotes"):
        table.add_group_column("g", rules={"marks > '50'": "a"})
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        table.add_group_column("g", rules={"mark > 50": "a"})
    with pytest.raises(SimplydataError, match="is not a condition"):
        table.add_group_column("g", rules={"marks": "a"})
    with pytest.raises(SimplydataError, match="all be text or all be numbers"):
        table.add_group_column("g", rules={"marks > 50": "a"}, otherwise=0)
    with pytest.raises(SimplydataError, match="not both"):
        table.add_group_column("g", from_column="marks", rules={"marks > 50": "a"})
    with pytest.raises(SimplydataError, match="otherwise= goes with rules="):
        table.add_group_column("g", from_column="marks", cut_at=[50], names=["a", "b"], otherwise="c")
    with pytest.raises(SimplydataError, match="pair each condition"):
        table.add_group_column("g", rules=["marks > 50"])
    with pytest.raises(SimplydataError, match="comparison simplydata does not know"):
        table.add_group_column("g", rules={"class in ['A']": "a"})


def test_formula_with_a_condition_points_to_rules(table):
    with pytest.raises(SimplydataError, match="rules="):
        table.add_column("x", "marks > 50")


def test_cut_at_form_still_works(table, marks):
    table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])
    expected = pd.cut(marks["marks"], [-np.inf, 40, 60, 80, np.inf], right=False, labels=["D", "C", "B", "A"])
    same_values(plain(table)["grade"], expected.astype(str))


# ---------------------------------------------------------------------------- rolling, previous, change
def shuffled_series(seed=3):
    rng = np.random.default_rng(seed)
    rows = [{"site": s, "day": d, "cases": float(rng.integers(0, 20))} for s in ("A", "B") for d in range(1, 9)]
    data = pd.DataFrame(rows)
    data.loc[5, "cases"] = np.nan
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("show,how", [("average", "mean"), ("total", "sum"), ("median", "median"),
                                      ("smallest", "min"), ("biggest", "max"), ("standard deviation", "std")])
def test_rolling_matches_pandas(engine, show, how, capsys):
    data = shuffled_series()
    t = build(data, engine)
    before = plain(t)
    t.add_rolling_column("r", from_column="cases", window=3, show=show, for_each="site", sorted_by="day")
    ordered = data.sort_values("day", kind="stable")
    expected = ordered.groupby("site")["cases"].transform(lambda s: getattr(s.rolling(3), how)()).sort_index()
    same_values(plain(t)["r"], expected)
    assert list(plain(t)["day"]) == list(data["day"])  # the rows keep their place
    assert f'Each "r" is the {show} of cases in this row and the 2 rows before it' in capsys.readouterr().out
    undo_gives_back(t, before)


def test_rolling_without_groups_and_order(engine):
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0]})
    t = build(data, engine)
    t.add_rolling_column("avg", from_column="x", window=2)
    same_values(plain(t)["avg"], data["x"].rolling(2).mean())
    t.add_rolling_column("big", from_column="x", window=2, show="biggest")
    same_values(plain(t)["big"], data["x"].rolling(2).max())
    t.add_rolling_column("same", from_column="x", window=1)
    same_values(plain(t)["same"], data["x"])


def test_rolling_errors(table):
    with pytest.raises(SimplydataError, match="not ready yet"):
        table.add_rolling_column("r", from_column="marks", window="7 days")
    with pytest.raises(SimplydataError, match="whole number of rows"):
        table.add_rolling_column("r", from_column="marks", window=0)
    with pytest.raises(SimplydataError, match='Did you mean show="median"'):
        table.add_rolling_column("r", from_column="marks", show="medain")
    with pytest.raises(SimplydataError, match="cannot do sums"):
        table.add_rolling_column("r", from_column="name")
    with pytest.raises(SimplydataError, match="at least two values"):
        table.add_rolling_column("r", from_column="marks", window=1, show="standard deviation")
    with pytest.raises(SimplydataError, match='Did you mean "test"'):
        table.add_rolling_column("r", from_column="marks", sorted_by="tst")
    with pytest.raises(SimplydataError, match="no option called every"):
        table.add_rolling_column("r", from_column="marks", every=3)


def test_previous_value_matches_shift(engine):
    data = shuffled_series()
    data["note"] = [f"n{i}" for i in range(len(data))]
    t = build(data, engine)
    before = plain(t)
    t.add_previous_value_column("before", from_column="cases", for_each="site", sorted_by="day")
    ordered = data.sort_values("day", kind="stable")
    same_values(plain(t)["before"], ordered.groupby("site")["cases"].shift(1).sort_index())
    t.add_previous_value_column("two_back", from_column="note", rows_back=2, for_each="site", sorted_by="day")
    same_values(plain(t)["two_back"], ordered.groupby("site")["note"].shift(2).sort_index())
    t.add_previous_value_column("plain", from_column="cases")
    same_values(plain(t)["plain"], data["cases"].shift(1))
    t.undo().undo().undo()
    pd.testing.assert_frame_equal(plain(t), before, check_dtype=False)
    with pytest.raises(SimplydataError, match="whole number of rows"):
        t.add_previous_value_column("p", from_column="cases", rows_back=0)


def test_change_matches_diff_and_pct_change(engine, capsys):
    data = shuffled_series()
    data.loc[data.index[2], "cases"] = 0.0
    t = build(data, engine)
    t.add_change_column("change", from_column="cases", for_each="site", sorted_by="day")
    ordered = data.sort_values("day", kind="stable")
    g = ordered.groupby("site")["cases"]
    same_values(plain(t)["change"], g.diff().sort_index())
    t.add_change_column("growth", from_column="cases", in_percent=True, for_each="site", sorted_by="day")
    previous = g.shift(1)
    expected = ((ordered["cases"] - previous) / previous * 100).replace([np.inf, -np.inf], np.nan).sort_index()
    same_values(plain(t)["growth"], expected)
    assert "change in cases from the row before in percent (separately for each site), in order of day" in \
        capsys.readouterr().out
    with pytest.raises(SimplydataError, match="True or False"):
        t.add_change_column("g", from_column="cases", in_percent="yes")
    with pytest.raises(SimplydataError, match="no option called as_percent"):
        t.add_change_column("g", from_column="cases", as_percent=True)


def test_change_on_example_growth(engine):
    import simplydata as sdata

    data = sdata.example("growth").sample(frac=1, random_state=1).reset_index(drop=True)
    t = build(data, engine)
    t.add_change_column("growth", from_column="population", in_percent=True, for_each="country", sorted_by="year")
    expected = data.sort_values("year", kind="stable").groupby("country")["population"].pct_change().sort_index() * 100
    same_values(plain(t)["growth"], expected)
