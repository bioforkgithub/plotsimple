"""Tests for the v0.3.x summary and test words: summary_of, every=, across=, percent_of,
count_different_values, correlation tables, compare_shape_to, add_adjusted_p_value_column,
table.column, and for_each= (formerly between=) on the tests.

Every number is checked against plain pandas, NumPy, SciPy or R, on both engines.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.numbers import Numbers
from simplydata.table import table as make_table
from simplydata.table.stat_tests import _adjust_p

ENGINES = ["pandas", "polars"]


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


def frame(t):
    return t._frame.to_pandas()


def snapshot(t):
    return frame(t).copy()


def unchanged(t, before):
    pd.testing.assert_frame_equal(frame(t), before)
    assert t._history == []


@pytest.fixture
def gappy():
    """Marks with empty cells, a group with one number, a group with no numbers, an empty class."""
    return pd.DataFrame({
        "class": ["A", "A", "A", "B", "B", "B", "C", "C", None, "D"],
        "subject": ["maths", "art", "maths", "art", "maths", "art", "maths", "art", "maths", "art"],
        "marks": [50.0, np.nan, 70.0, 40.0, 80.0, 65.0, 90.0, np.nan, 60.0, np.nan],
        "hours": [1.0, 2.0, 3.0, 2.0, 5.0, 1.0, 6.0, 4.0, np.nan, 2.0],
        "name": ["a", "b", "b", "c", None, " ", "d", "d", "e", "f"],
    })


@pytest.fixture
def daily():
    """60 days of sales over Jan-Mar 2026, two shops, a few empty dates."""
    rng = np.random.default_rng(7)
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    data = pd.DataFrame({
        "date": list(days) * 2,
        "shop": ["North"] * 60 + ["South"] * 60,
        "sold": rng.integers(0, 50, 120).astype(float),
    })
    data.loc[[5, 70], "date"] = pd.NaT
    return data


# ================================================================ for_each= on the tests
def test_tests_take_for_each(engine, marks, capsys):
    t = make(marks, engine)
    a = t.compare_averages("marks", for_each="class")._frame.to_pandas()
    out = capsys.readouterr().out
    assert "Note" not in out.splitlines()[0]
    want = stats.ttest_ind(marks.loc[marks["class"] == "A", "marks"], marks.loc[marks["class"] == "B", "marks"],
                           equal_var=False)
    assert a["p-value"][0] == pytest.approx(want.pvalue)
    pairs = t.compare_each_pair("marks", for_each="test")._frame.to_pandas()
    assert len(pairs) == 3
    capsys.readouterr()
    data = marks.assign(result=np.where(marks["marks"] >= 60, "pass", "fail"))
    counts = make(data, engine).compare_counts("result", for_each="class")._frame.to_pandas()
    observed = pd.crosstab(data["result"], data["class"]).to_numpy()
    assert counts["p-value"][0] == pytest.approx(stats.chi2_contingency(observed).pvalue)


@pytest.mark.parametrize("word", ["compare_averages", "compare_each_pair"])
def test_between_still_works_with_one_note(engine, marks, capsys, word):
    t = make(marks, engine)
    old = getattr(t, word)("marks", between="test")._frame.to_pandas()
    out = capsys.readouterr().out.splitlines()
    assert out[0] == 'Note: "between=" is now called "for_each=". The old name will stop working in a later version.'
    new = getattr(t, word)("marks", for_each="test")._frame.to_pandas()
    assert "Note" not in capsys.readouterr().out
    pd.testing.assert_frame_equal(old, new)
    getattr(t, word)("marks", between="test")
    assert "is now called" not in capsys.readouterr().out  # once per session


def test_between_on_compare_counts_and_errors(engine, marks, capsys):
    data = marks.assign(result=np.where(marks["marks"] >= 60, "pass", "fail"))
    t = make(data, engine)
    t.compare_counts("result", between="class")
    assert '"between=" is now called "for_each="' in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="only once"):
        t.compare_counts("result", between="class", for_each="class")
    with pytest.raises(SimplydataError, match="for_each= as a column name"):
        t.compare_averages("marks")
    with pytest.raises(SimplydataError, match="the for_each= column"):
        t.compare_averages("marks", for_each="marks")


def test_messages_name_for_each(marks, capsys):
    make_table(marks).compare_averages("marks", for_each="test")
    out = capsys.readouterr().out
    assert 'table.compare_each_pair("marks", for_each="test")' in out and "between=" not in out
    make_table(marks).compare_each_pair("marks", for_each="test")
    assert 'Each pair of "test" groups' in capsys.readouterr().out


# ================================================================ summary_of
def test_summary_of_matches_pandas(engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    got = frame(t.summary_of(["marks", "hours"], for_each="class",
                             show=["count", "average", "median", "total", "smallest", "biggest",
                                   "standard deviation", "variance", "standard error", "range",
                                   "interquartile range", "empty cells"]))
    assert list(got.columns[:2]) == ["column", "class"]
    for column in ("marks", "hours"):
        part = got[got["column"] == column].reset_index(drop=True)
        groups = gappy.groupby("class", dropna=False, sort=True)[column]
        want = pd.DataFrame({
            "count": groups.count(), "average": groups.mean(), "median": groups.median(), "total": groups.sum(),
            "smallest": groups.min(), "biggest": groups.max(), "standard deviation": groups.std(ddof=1),
            "variance": groups.var(ddof=1), "standard error": groups.std(ddof=1) / np.sqrt(groups.count()),
            "range": groups.max() - groups.min(),
            "interquartile range": groups.quantile(0.75) - groups.quantile(0.25),
            "empty cells": groups.apply(lambda s: s.isna().sum()),
        }).reset_index()
        assert part["class"].tolist()[:4] == ["A", "B", "C", "D"] and pd.isna(part["class"].tolist()[4])
        for name in want.columns[1:]:
            np.testing.assert_allclose(part[name].astype(float), want[name].astype(float), equal_nan=True,
                                       err_msg=f"{column} {name}")
    assert got["count"].dtype.kind == "i" and got["empty cells"].dtype.kind == "i"
    out = capsys.readouterr().out
    assert out.splitlines()[0] == 'Note: empty cells were skipped: 3 in "marks", 1 in "hours".'
    assert 'Summary of "marks" and "hours" for each "class":' in out
    assert '  "marks" / A: count 2, average 60, median 60, total 120' in out
    unchanged(t, before)


def test_summary_of_confidence_interval_and_default(engine, marks, capsys):
    t = make(marks, engine)
    got = frame(t.summary_of("marks", for_each="class", show=["average", "ci"]))
    for i, cls in enumerate(["A", "B"]):
        v = marks.loc[marks["class"] == cls, "marks"]
        low, high = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
        assert got["average 95 % CI low"][i] == pytest.approx(low)
        assert got["average 95 % CI high"][i] == pytest.approx(high)
    assert "A: average" in capsys.readouterr().out
    plain = frame(t.summary_of("marks"))
    assert list(plain.columns) == ["column", "count", "average", "standard deviation", "smallest", "median", "biggest"]
    assert plain["standard deviation"][0] == pytest.approx(marks["marks"].std())
    assert "95 % CI" not in capsys.readouterr().out


def test_summary_of_own_measure_and_aliases(engine, marks, capsys):
    t = make(marks, engine)

    def spread(values):
        return values.max() - values.min()

    got = frame(t.summary_of("marks", for_each="class", show=["mean", "sd", spread, lambda v: len(v)]))
    assert list(got.columns) == ["column", "class", "average", "standard deviation", "spread", "own measure"]
    want = marks.groupby("class")["marks"].agg(lambda s: s.max() - s.min())
    np.testing.assert_allclose(got["spread"], want.to_numpy())
    assert got["own measure"].tolist() == [12.0, 12.0]
    with pytest.raises(SimplydataError, match='Your measure "broken" did not work on "marks" in group A'):
        def broken(values):
            raise ValueError("no")
        t.summary_of("marks", for_each="class", show=[broken])


def test_summary_of_errors(engine, gappy):
    t = make(gappy, engine)
    with pytest.raises(SimplydataError, match='Did you mean show="average"'):
        t.summary_of("marks", show=["averge"])
    with pytest.raises(SimplydataError, match='"name" has text'):
        t.summary_of("name")
    with pytest.raises(SimplydataError, match="needs the column"):
        t.summary_of()
    with pytest.raises(SimplydataError, match="at least one measure"):
        t.summary_of("marks", show=[])
    with pytest.raises(SimplydataError, match='called "column"'):
        make(gappy.rename(columns={"class": "column"}), engine).summary_of("marks", for_each="column")
    with pytest.raises(SimplydataError, match="no option called across="):
        t.summary_of("marks", for_each="class", across="subject")


# ================================================================ every=
@pytest.mark.parametrize("every, period", [("month", "M"), ("week", "W-SUN"), ("year", "Y"), ("quarter", "Q"),
                                           ("day", "D")])
def test_every_matches_pandas_periods(engine, daily, every, period, capsys):
    t = make(daily, engine)
    before = snapshot(t)
    got = frame(t.total_of("sold", for_each=["shop", "date"], every=every))
    kept = daily.dropna(subset=["date"])
    want = kept.groupby(["shop", kept["date"].dt.to_period(period)])["sold"].sum().reset_index()
    starts = want["date"].map(lambda p: p.start_time).to_numpy(dtype="datetime64[ns]")
    real = got.dropna(subset=["date"]).reset_index(drop=True)
    assert real["shop"].tolist() == want["shop"].tolist()
    np.testing.assert_array_equal(pd.to_datetime(real["date"]).to_numpy(dtype="datetime64[ns]"), starts)
    np.testing.assert_allclose(real["total of sold"], want["sold"])
    empty = got[got["date"].isna()]
    assert empty["total of sold"].tolist() == daily[daily["date"].isna()].groupby("shop")["sold"].sum().tolist()
    out = capsys.readouterr().out
    assert f'for each "shop" and {every} of "date"' in out
    assert f"Note: in the new table each {every} is shown by" in out
    unchanged(t, before)


def test_every_labels_and_dates(engine, daily, capsys):
    t = make(daily.assign(date=daily["date"].dt.date), engine)  # plain dates, not date-times
    if engine == "pandas":
        t.make_dates("date")
    capsys.readouterr()
    got = frame(t.average_of("sold", for_each="date", every="month"))
    assert len(got) == 4  # Jan, Feb, Mar and the empty dates
    out = capsys.readouterr().out
    assert 'The average of "sold" for each month of "date": 2026-01 ' in out and "(empty)" in out
    t.count_rows(for_each="date", every="week")
    assert "week of 2025-12-29" in capsys.readouterr().out  # 1 January 2026 is a Thursday


def test_every_on_other_words(engine, daily):
    t = make(daily, engine)
    kept = daily.dropna(subset=["date"])
    months = kept["date"].dt.to_period("M")
    rows = frame(t.count_rows(for_each="date", every="month")).dropna()
    assert rows["count"].tolist() == kept.groupby(months).size().tolist()
    summary = frame(t.summary_of("sold", for_each="date", every="month", show=["median"])).dropna(subset=["date"])
    np.testing.assert_allclose(summary["median"], kept.groupby(months)["sold"].median())
    different = frame(t.count_different_values("sold", for_each="date", every="month")).dropna()
    assert different["different values"].tolist() == kept.groupby(months)["sold"].nunique().tolist()
    share = frame(t.percent_of("sold", above=25, for_each="date", every="month")).dropna()
    np.testing.assert_allclose(share["percent"], kept.groupby(months)["sold"].apply(lambda s: (s > 25).mean() * 100))


def test_every_errors(engine, daily, marks):
    t = make(daily, engine)
    with pytest.raises(SimplydataError, match="needs the column of dates in for_each="):
        t.total_of("sold", every="month")
    with pytest.raises(SimplydataError, match='Did you mean every="month"'):
        t.total_of("sold", for_each="date", every="mnth")
    with pytest.raises(SimplydataError, match=r'make them dates first: table.make_dates\("shop"\)'):
        t.total_of("sold", for_each="shop", every="month")
    with pytest.raises(SimplydataError, match="needs the column of dates"):
        t.count_rows(every="month")


# ================================================================ across=
def test_across_matches_pivot_table(engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    got = frame(t.average_of("marks", for_each="class", across="subject"))
    want = gappy.pivot_table(index="class", columns="subject", values="marks", aggfunc="mean", dropna=False)
    assert list(got.columns) == ["class", "art", "maths"]
    assert got["class"].tolist()[:4] == ["A", "B", "C", "D"] and pd.isna(got["class"].tolist()[4])
    for name in ("art", "maths"):
        np.testing.assert_allclose(got[name][:4].astype(float), want[name].reindex(["A", "B", "C", "D"]).astype(float),
                                   equal_nan=True)
    assert got["maths"][4] == 60  # the empty class keeps its row
    out = capsys.readouterr().out
    assert 'for each "class" (rows) and each "subject" (columns): a new table with 5 rows' in out
    assert "  A: art empty, maths 60" in out
    total = frame(t.total_of("hours", for_each="subject", across="class"))
    assert list(total.columns) == ["subject", "A", "B", "C", "D", "(empty)"]
    unchanged(t, before)


def test_count_rows_across_is_crosstab(engine, marks, capsys):
    data = marks.assign(result=np.where(marks["marks"] >= 60, "pass", "fail"))
    t = make(data, engine)
    got = frame(t.count_rows(for_each="class", across="result"))
    want = pd.crosstab(data["class"], data["result"])
    assert got["fail"].tolist() == want["fail"].tolist() and got["pass"].tolist() == want["pass"].tolist()
    assert got["fail"].dtype.kind == "i"
    shares = frame(t.count_rows(for_each="class", across="result", in_percent=True))
    want = pd.crosstab(data["class"], data["result"], normalize="index") * 100
    np.testing.assert_allclose(shares[["fail", "pass"]], want[["fail", "pass"]])
    assert "each row adds up to 100 %" in capsys.readouterr().out
    missing = frame(make(pd.DataFrame({"g": ["a", "a", "b"], "h": ["x", "y", "x"]}), engine)
                    .count_rows(for_each="g", across="h"))
    assert missing["y"].tolist() == [1, 0]  # a missing pair counts 0
    part = frame(t.count_rows(for_each="class", in_percent=True))
    assert part["percent"].tolist() == [50.0, 50.0]


def test_across_errors(engine, marks):
    t = make(marks, engine)
    with pytest.raises(SimplydataError, match="for_each= gives its rows"):
        t.average_of("marks", across="class")
    with pytest.raises(SimplydataError, match="cannot be both"):
        t.average_of("marks", for_each="class", across="class")
    with pytest.raises(SimplydataError, match="one column"):
        t.average_of(["marks", "height"], for_each="class", across="test")
    with pytest.raises(SimplydataError, match="Did you mean"):
        t.count_rows(for_each="class", across="tset")
    with pytest.raises(SimplydataError, match="needs groups"):
        t.count_rows(in_percent=True)


# ================================================================ percent_of
def test_percent_of_condition_matches_pandas_and_r(engine, marks, capsys):
    t = make(marks, engine)
    before = snapshot(t)
    share = t.percent_of("marks", above=60)
    assert isinstance(share, float) and share == pytest.approx((marks["marks"] > 60).mean() * 100)
    got = frame(t.percent_of("marks", at_least=60, for_each="class"))
    want = marks.groupby("class")["marks"].apply(lambda s: (s >= 60).mean() * 100)
    np.testing.assert_allclose(got["percent"], want)
    assert got["rows"].tolist() == [12, 12]
    for i, row in got.iterrows():
        ci = stats.binomtest(int(row["matching rows"]), 12).proportion_ci(method="exact")
        assert row["percent 95 % CI low"] == pytest.approx(ci.low * 100)
        assert row["percent 95 % CI high"] == pytest.approx(ci.high * 100)
    out = capsys.readouterr().out
    assert "exact Clopper-Pearson interval" in out and 'Percent of rows where "marks" is at least 60, for each "class"' in out
    unchanged(t, before)


def test_percent_ci_matches_r(engine):
    # R 4.6.2: binom.test(14, 18)$conf.int -> 0.523627234263520 0.935907952282333
    #          binom.test(0, 7)$conf.int   -> 0 0.409616397225003
    data = pd.DataFrame({"g": ["a"] * 18 + ["b"] * 7, "v": [1.0] * 14 + [0.0] * 4 + [0.0] * 7})
    got = frame(make(data, engine).percent_of("v", equal_to=1, for_each="g"))
    assert got["percent 95 % CI low"].tolist() == pytest.approx([52.3627234263520, 0.0])
    assert got["percent 95 % CI high"].tolist() == pytest.approx([93.5907952282333, 40.9616397225003])


def test_percent_of_empty_cells_and_text(engine, gappy, capsys):
    t = make(gappy, engine)
    assert t.percent_of("marks", above=55) == pytest.approx(5 / 7 * 100)
    assert capsys.readouterr().out.startswith('Note: 3 rows with an empty cell in "marks" were left out of the percent.')
    assert t.percent_of("marks", is_empty=True) == pytest.approx(30.0)
    assert t.percent_of("subject", equal_to="art") == pytest.approx(50.0)
    by = frame(t.percent_of("marks", above=55, for_each="class"))
    assert by["rows"].tolist() == [2, 3, 1, 0, 1]
    assert math.isnan(by["percent"][3])  # class D has no numbers: no percent


def test_percent_of_shares(engine, capsys):
    fruit = sdata.example("fruit")
    t = make(fruit, engine)
    total = frame(t.percent_of("count", for_each="shop"))
    want = fruit.groupby("shop")["count"].sum() / fruit["count"].sum() * 100
    np.testing.assert_allclose(total["percent"], want)
    assert total["total of count"].tolist() == fruit.groupby("shop")["count"].sum().tolist()
    assert "Share of the total of" in capsys.readouterr().out
    rows = frame(t.percent_of(for_each="fruit"))
    np.testing.assert_allclose(rows["percent"], fruit.groupby("fruit").size() / len(fruit) * 100)
    assert {"count", "percent 95 % CI low", "percent 95 % CI high"} <= set(rows.columns)


def test_percent_of_errors(engine, marks):
    t = make(marks, engine)
    with pytest.raises(SimplydataError, match="needs a column with a condition"):
        t.percent_of()
    with pytest.raises(SimplydataError, match="needs for_each= to share out"):
        t.percent_of("marks")
    with pytest.raises(SimplydataError, match="one condition at a time"):
        t.percent_of("marks", above=50, below=80)
    with pytest.raises(SimplydataError, match="compares numbers or dates"):
        t.percent_of("class", above=3)
    with pytest.raises(SimplydataError, match="needs the column name first"):
        t.percent_of(above=60)
    with pytest.raises(SimplydataError, match="cannot be both"):
        t.percent_of("class", equal_to="A", for_each="class")


# ================================================================ count_different_values
def test_count_different_values_matches_nunique(engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    n = t.count_different_values("name")
    cleaned = gappy["name"].where(gappy["name"].str.strip() != "")
    assert n == cleaned.nunique() == 6
    out = capsys.readouterr().out
    assert out.splitlines() == ['Note: 2 empty cells in "name" were not counted.', '"name" has 6 different values.']
    got = frame(t.count_different_values("name", for_each="class"))
    want = cleaned.groupby(gappy["class"], dropna=False).nunique()
    assert got["different values"].tolist() == [2, 1, 1, 1, 1] == [want["A"], want["B"], want["C"], want["D"],
                                                                    want[np.nan]]
    assert frame(t.count_different_values("marks", for_each="subject"))["different values"].tolist() == [2, 5]
    assert gappy.groupby("subject")["marks"].nunique().tolist() == [2, 5]
    unchanged(t, before)
    with pytest.raises(SimplydataError, match="one column name"):
        t.count_different_values(["name"])


# ================================================================ correlation tables
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_table_matches_pandas(engine, gappy, method, capsys):
    data = gappy.assign(extra=[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, np.nan])
    t = make(data, engine)
    before = snapshot(t)
    names = ["marks", "hours", "extra"]
    got = frame(t.correlation_between(*names, method=method))
    want = data[names].corr(method=method)  # pairwise, like simplydata
    assert got["column"].tolist() == names
    np.testing.assert_allclose(got[names].to_numpy(), want.to_numpy())
    p = frame(t.correlation_between(names, method=method, show="p-value"))
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    both = data[["marks", "extra"]].dropna()
    assert p["extra"][0] == pytest.approx(fn(both["marks"], both["extra"])[1])
    assert pd.isna(p["marks"][0])
    pairs = frame(t.correlation_between(*names, method=method, show="each pair"))
    assert pairs[["column 1", "column 2"]].values.tolist() == [["marks", "hours"], ["marks", "extra"],
                                                                ["hours", "extra"]]
    assert pairs["rows"].tolist() == [6, 7, 8]
    assert pairs["r"][2] == pytest.approx(want.loc["hours", "extra"])
    out = capsys.readouterr().out
    assert "each pair uses the rows where both columns have a number" in out
    assert "not adjusted for testing 3 pairs" in out
    unchanged(t, before)


def test_correlation_table_prints_strongest(marks, capsys):
    data = marks.assign(double=marks["marks"] * 2 + 1, minus=-marks["height"])
    make_table(data).correlation_between("marks", "height", "double", "minus", "test")
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith('  "marks" and "double": r = 1.00') and out[2].startswith('  "height" and "minus": r = -1.00')
    assert out[4] == "  ... and 7 more pairs (the new table has them all)."


def test_correlation_two_columns_unchanged(engine, marks, capsys):
    r = make(marks, engine).correlation_between(["marks", "height"])
    assert r == pytest.approx(stats.pearsonr(marks["marks"], marks["height"])[0])
    assert capsys.readouterr().out.count("\n") == 1


def test_correlation_table_errors(engine, marks):
    t = make(marks, engine)
    with pytest.raises(SimplydataError, match="three or more columns"):
        t.correlation_between("marks", "height", show="p-value")
    with pytest.raises(SimplydataError, match='"marks" is listed twice'):
        t.correlation_between("marks", "height", "marks")
    with pytest.raises(SimplydataError, match='Did you mean show="each pair"'):
        t.correlation_between("marks", "height", "test", show="each pairs")
    with pytest.raises(SimplydataError, match="two column names"):
        t.correlation_between("marks")
    with pytest.raises(SimplydataError, match='"name" has text'):
        t.correlation_between("marks", "height", "name")


# ================================================================ compare_shape_to
def test_compare_shape_to_matches_r(engine, capsys):
    # R 4.6.2: shapiro.test(x) -> W = 0.984911223630307, p = 0.981003667441439
    #          shapiro.test(y) -> W = 0.610215299500653, p = 7.32220250835809e-05
    x = [61, 66, 77, 73, 98, 55, 42, 88, 70, 64, 59, 81, 93, 47, 52, 68, 74, 79, 85, 90]
    y = [1, 1.2, 1.1, 1.3, 1.25, 5, 9, 1.05, 1.15, 20]
    t = make(pd.DataFrame({"v": x + y, "g": ["x"] * 20 + ["y"] * 10}), engine)
    before = snapshot(t)
    got = frame(t.compare_shape_to("v", "normal", for_each="g"))
    assert got["statistic"].tolist() == pytest.approx([0.984911223630307, 0.610215299500653], rel=1e-6)
    assert got["p-value"].tolist() == pytest.approx([0.981003667441439, 7.32220250835809e-05], rel=1e-4)
    assert got["rows"].tolist() == [20, 10] and got["test"][0] == "Shapiro-Wilk test"
    out = capsys.readouterr().out
    assert "  x: could well be bell-shaped (W = 0.985, p = 0.981, 20 values)" in out
    assert "  y: do not look bell-shaped (W = 0.610, p < 0.0001, 10 values)" in out
    one = frame(make(pd.DataFrame({"v": y}), engine).compare_shape_to("v"))
    assert one["p-value"][0] == pytest.approx(stats.shapiro(y).pvalue)
    assert "do not look bell-shaped (normal)" in capsys.readouterr().out
    unchanged(t, before)


def test_compare_shape_to_small_groups_and_errors(engine, gappy, capsys):
    t = make(gappy, engine)
    got = frame(t.compare_shape_to("marks", for_each="class"))
    assert got["rows"].tolist() == [2, 3, 1, 0, 1]
    assert math.isnan(got["p-value"][0]) and not math.isnan(got["p-value"][1])
    assert "A: not tested (only 2 values: Shapiro-Wilk needs at least 3)." in capsys.readouterr().out
    with pytest.raises(SimplydataError, match='Did you mean shape="normal"'):
        t.compare_shape_to("marks", "normall")
    with pytest.raises(SimplydataError, match="every value is the same"):
        make(pd.DataFrame({"v": [1.0, 1.0, 1.0, 1.0]}), engine).compare_shape_to("v")
    with pytest.raises(SimplydataError, match="needs at least 3"):
        make(pd.DataFrame({"v": [1.0, 2.0]}), engine).compare_shape_to("v")
    with pytest.raises(SimplydataError, match='"name" has text'):
        t.compare_shape_to("name")


def test_compare_shape_to_large_sample_note(capsys):
    rng = np.random.default_rng(3)
    make_table(pd.DataFrame({"v": rng.exponential(size=200)})).compare_shape_to("v")
    assert "central limit theorem" in capsys.readouterr().out


# ================================================================ add_adjusted_p_value_column
# R 4.6.2:  p <- c(0.01, 0.04, 0.03, 0.005, 0.2, 0.04, NA, 0.5); p.adjust(p, method = m)
R_ADJUSTED = {
    "holm": [0.060, 0.160, 0.150, 0.035, 0.400, 0.160, None, 0.500],
    "bonferroni": [0.070, 0.280, 0.210, 0.035, 1.000, 0.280, None, 1.000],
    "benjamini-hochberg": [0.035, 0.056, 0.056, 0.035, 0.233333333333333, 0.056, None, 0.5],
    "benjamini-yekutieli": [0.09075, 0.14520, 0.14520, 0.09075, 0.60500, 0.14520, None, 1.00000],
}
P_VALUES = [0.01, 0.04, 0.03, 0.005, 0.2, 0.04, np.nan, 0.5]


@pytest.mark.parametrize("method", list(R_ADJUSTED))
def test_adjusted_p_values_match_r(engine, method, capsys):
    t = make(pd.DataFrame({"test": list("abcdefgh"), "p-value": P_VALUES}), engine)
    t.add_adjusted_p_value_column("q", method=method)
    got = frame(t)["q"].tolist()
    want = R_ADJUSTED[method]
    for g, w in zip(got, want):
        assert (w is None and pd.isna(g)) or g == pytest.approx(w)
    out = capsys.readouterr().out
    assert 'Added "q": 7 p-values from "p-value" adjusted with' in out
    assert "1 empty cell" in out
    t.undo()
    assert "q" not in t.column_names()


def test_adjusted_p_value_defaults_and_aliases(engine, capsys):
    t = make(pd.DataFrame({"p": [0.01, 0.02, 0.03]}), engine)
    t.add_adjusted_p_value_column(from_column="p")
    assert frame(t)["p (Holm)"].tolist() == pytest.approx([0.03, 0.04, 0.04])
    t.add_adjusted_p_value_column("q", from_column="p", method="fdr")
    assert frame(t)["q"].tolist() == pytest.approx([0.03, 0.03, 0.03])
    t.add_adjusted_p_value_column("q2", from_column="p", method="Benjamini-Hochberg")
    assert frame(t)["q2"].tolist() == pytest.approx([0.03, 0.03, 0.03])
    assert t.steps() is t and len(t._history) == 3
    assert 'as R p.adjust(method = "BH")' in capsys.readouterr().out


def test_adjusted_p_values_on_test_results(engine, marks):
    results = make(marks, engine).compare_each_pair("marks", for_each="test")
    results.add_adjusted_p_value_column("again", method="holm")
    got = frame(results)
    np.testing.assert_allclose(got["again"], got["p-value (Holm)"])  # the same Holm as compare_each_pair
    pairs = make(marks, engine).correlation_between("marks", "height", "test", show="each pair")
    pairs.add_adjusted_p_value_column("q", method="bh")
    np.testing.assert_allclose(frame(pairs)["q"], stats.false_discovery_control(frame(pairs)["p-value"]))


def test_adjust_p_against_scipy_on_random_values():
    p = np.random.default_rng(5).uniform(size=50) ** 3
    np.testing.assert_allclose(_adjust_p(p, "benjamini-hochberg"), stats.false_discovery_control(p, method="bh"))
    np.testing.assert_allclose(_adjust_p(p, "benjamini-yekutieli"), stats.false_discovery_control(p, method="by"))


def test_adjusted_p_value_errors(engine, marks):
    t = make(pd.DataFrame({"p": [0.2, 1.5], "name": ["a", "b"]}), engine)
    with pytest.raises(SimplydataError, match="go from 0 to 1, but it has 1.50"):
        t.add_adjusted_p_value_column("q", from_column="p")
    with pytest.raises(SimplydataError, match='Did you mean method="holm"'):
        t.add_adjusted_p_value_column("q", from_column="p", method="hollm")
    with pytest.raises(SimplydataError, match='Column "p-value" not found'):
        t.add_adjusted_p_value_column("q")
    with pytest.raises(SimplydataError, match='"name" has text'):
        t.add_adjusted_p_value_column("q", from_column="name")
    with pytest.raises(SimplydataError, match="every cell is empty"):
        make(pd.DataFrame({"p": [np.nan, np.nan]}), engine).add_adjusted_p_value_column("q", from_column="p")
    assert t._history == []


# ================================================================ table.column
def test_column_gives_numbers(engine, gappy, marks, capsys):
    t = make(marks, engine)
    before = snapshot(t)
    m = t.column("marks")
    assert isinstance(m, Numbers)
    np.testing.assert_array_equal(m.as_numpy(), marks["marks"].to_numpy())
    whole = t.column("test")
    assert whole.as_numpy().dtype.kind == "i" and whole.as_numpy().tolist() == marks["test"].tolist()
    assert capsys.readouterr().out == ""
    unchanged(t, before)
    gaps = make(gappy, engine).column("marks").as_numpy()
    np.testing.assert_array_equal(np.isnan(gaps), gappy["marks"].isna().to_numpy())
    assert make(gappy, engine).column("marks").average() == pytest.approx(gappy["marks"].mean())
    names = make(gappy, engine).column("name")
    assert names[:3] == ["a", "b", "b"] and names[4] is None
    assert 'holds text, not numbers, so it comes back as a plain Python list' in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="one column name in quotes"):
        t.column(["marks"])
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        t.column("mark")


def test_example_tables_run(capsys):
    t = sdata.table(sdata.example("marks"))
    t.summary_of("marks", for_each="class", show=["average", "standard deviation", "count"])
    t.average_of("marks", for_each="class", across="subject")
    t.percent_of("marks", above=60, for_each="class")
    t.count_different_values("name", for_each="class")
    t.compare_shape_to("marks", "normal", for_each="class")
    results = t.correlation_between("hours", "marks", "height", show="each pair")
    results.add_adjusted_p_value_column("q", method="benjamini-hochberg")
    assert t.column("marks").average() == pytest.approx(sdata.example("marks")["marks"].mean())
    g = sdata.table(sdata.example("growth"))
    g.make_dates("year")
    g.average_of("population", for_each="year", every="year")
    sdata.table(sdata.example("fruit")).count_rows(for_each="shop", across="fruit")
    assert capsys.readouterr().out
