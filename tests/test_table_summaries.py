"""Tests for the summary words: numbers checked against plain pandas, NumPy and SciPy."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.summaries import GROUPS_IN_SENTENCE, _p_words, _say_number, _strength

ENGINES = ["pandas", "polars"]

WORDS = {
    "average_of": ("average", lambda s: s.mean()),
    "median_of": ("median", lambda s: s.median()),
    "total_of": ("total", lambda s: s.sum()),
    "smallest_of": ("smallest", lambda s: s.min()),
    "biggest_of": ("biggest", lambda s: s.max()),
    "standard_deviation_of": ("standard deviation", lambda s: s.std(ddof=1)),
}


@pytest.fixture
def gappy():
    """Marks with empty cells, including a group with only one number and an empty class."""
    return pd.DataFrame({
        "class": ["A", "A", "A", "B", "B", "B", "C", "C", None],
        "subject": ["maths", "art", "maths", "art", "maths", "art", "maths", "art", "maths"],
        "marks": [50.0, np.nan, 70.0, 40.0, 80.0, np.nan, 90.0, np.nan, 60.0],
        "hours": [1.0, 2.0, 3.0, 2.0, 5.0, 1.0, 6.0, 4.0, np.nan],
        "name": list("abcdefghi"),
        "passed": [True] * 9,
    })


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


def snapshot(t):
    return t._frame.to_pandas().copy()


def unchanged(t, before):
    pd.testing.assert_frame_equal(t._frame.to_pandas(), before)
    assert t._history == []


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


# ---------------------------------------------------------------- plain numbers
@pytest.mark.parametrize("word", list(WORDS))
def test_single_number_matches_pandas(word, engine, marks, capsys):
    t = make(marks, engine)
    before = snapshot(t)
    for col in ("marks", "height", "test"):
        got = getattr(t, word)(col)
        assert isinstance(got, (int, float)) and not isinstance(got, np.generic)
        assert got == pytest.approx(WORDS[word][1](marks[col]))
    out = capsys.readouterr().out
    assert "Note" not in out  # no empty cells: no note, only the sentence for each answer
    start = WORDS[word][0]
    assert out.splitlines()[0] == f'The {start} of "marks" is {_say_number(WORDS[word][1](marks["marks"]))} (24 values).'
    unchanged(t, before)


def test_standard_deviation_is_sample(engine, marks):
    t = make(marks, engine)
    assert t.standard_deviation_of("marks") == pytest.approx(np.std(marks["marks"].to_numpy(), ddof=1))
    assert t.standard_deviation_of("marks") != pytest.approx(np.std(marks["marks"].to_numpy(), ddof=0))


@pytest.mark.parametrize("word", list(WORDS))
def test_empty_cells_skipped_with_note(word, engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    got = getattr(t, word)("marks")
    assert got == pytest.approx(WORDS[word][1](gappy["marks"].dropna()))
    assert capsys.readouterr().out.startswith('Note: 3 empty cells in "marks" were skipped.\n')
    unchanged(t, before)


def test_one_empty_cell_note(engine, gappy, capsys):
    t = make(gappy, engine)
    t.average_of("hours")
    assert capsys.readouterr().out.startswith('Note: 1 empty cell in "hours" was skipped.\n')


def test_nan_in_polars_counts_as_empty():
    pl = pytest.importorskip("polars")
    t = make_table(pl.DataFrame({"x": [1.0, float("nan"), 3.0, None]}))
    assert t.average_of("x") == pytest.approx(2.0)


def test_all_empty_gives_nan(engine):
    data = pd.DataFrame({"g": ["a", "b"], "x": [np.nan, np.nan]})
    t = make(data, engine)
    assert math.isnan(t.average_of("x"))
    assert math.isnan(t.standard_deviation_of("x"))


# ---------------------------------------------------------------- for_each
@pytest.mark.parametrize("word", list(WORDS))
def test_for_each_matches_pandas_groupby(word, engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    start, calc = WORDS[word]
    summary = getattr(t, word)("marks", for_each="class")
    assert summary is not t and summary._is_simplydata_table
    got = summary._done()
    if engine == "polars":
        got = got.to_pandas()
    assert list(got.columns) == ["class", f"{start} of marks"]
    expected = gappy.groupby("class", dropna=False)["marks"].agg(calc)
    # sorted by group, empty group last
    assert got["class"].iloc[:3].tolist() == ["A", "B", "C"]
    assert pd.isna(got["class"].iloc[3])
    for key, value in zip(got["class"], got[f"{start} of marks"]):
        want = expected[np.nan] if pd.isna(key) else expected[key]
        if pd.isna(want):
            assert pd.isna(value)  # e.g. SD of a single number
        else:
            assert value == pytest.approx(want)
    assert capsys.readouterr().out.startswith('Note: 3 empty cells in "marks" were skipped.\n')
    unchanged(t, before)


def test_for_each_many_columns(engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    got = t.average_of(["marks", "hours"], for_each=["class", "subject"])._done()
    if engine == "polars":
        got = got.to_pandas()
    assert list(got.columns) == ["class", "subject", "average of marks", "average of hours"]
    expected = (gappy.groupby(["class", "subject"], dropna=False)[["marks", "hours"]].mean()
                .reset_index().sort_values(["class", "subject"], na_position="last").reset_index(drop=True))
    assert got["class"].tolist()[:-1] == expected["class"].tolist()[:-1]
    assert got["subject"].tolist() == expected["subject"].tolist()
    np.testing.assert_allclose(got["average of marks"], expected["marks"], equal_nan=True)
    np.testing.assert_allclose(got["average of hours"], expected["hours"], equal_nan=True)
    assert capsys.readouterr().out.startswith('Note: empty cells were skipped: 3 in "marks", 1 in "hours".\n')
    unchanged(t, before)


def test_list_without_for_each_gives_one_row_table(engine, marks):
    t = make(marks, engine)
    got = t.total_of(["marks", "height"])._done()
    if engine == "polars":
        got = got.to_pandas()
    assert list(got.columns) == ["total of marks", "total of height"]
    assert got["total of marks"].iloc[0] == pytest.approx(marks["marks"].sum())
    assert got["total of height"].iloc[0] == pytest.approx(marks["height"].sum())


def test_pandas_result_has_clean_index(marks):
    got = make_table(marks).average_of("marks", for_each="class")._done()
    assert got.index.tolist() == [0, 1]


def test_summary_works_with_chart_and_save(engine, marks, tmp_path):
    summary = make(marks, engine).average_of("marks", for_each="class")
    chart = summary.chart().bars(x="class", y="average of marks")
    assert len(chart._render().axes[0].patches) == 2
    summary.save(tmp_path / "summary.csv")
    back = pd.read_csv(tmp_path / "summary.csv")
    assert back["average of marks"].tolist() == pytest.approx(marks.groupby("class")["marks"].mean().tolist())


def test_engine_kept(engine, marks):
    t = make(marks, engine)
    assert t.average_of("marks", for_each="class")._engine == t._engine
    assert t.count_each("class")._engine == t._engine


# ---------------------------------------------------------------- counting
def test_count_rows(engine, gappy, capsys):
    t = make(gappy, engine)
    before = snapshot(t)
    assert t.count_rows() == 9 and isinstance(t.count_rows(), int)
    got = t.count_rows(for_each="class")._done()
    if engine == "polars":
        got = got.to_pandas()
    assert list(got.columns) == ["class", "count"]
    assert got["class"].tolist()[:3] == ["A", "B", "C"] and pd.isna(got["class"].iloc[3])
    assert got["count"].tolist() == [3, 3, 2, 1]
    two = t.count_rows(for_each=["class", "subject"])._done()
    if engine == "polars":
        two = two.to_pandas()
    assert two["count"].sum() == 9
    expected = gappy.groupby(["class", "subject"], dropna=False).size()
    assert len(two) == len(expected)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "The table has 9 rows."
    assert out[2] == 'Rows for each "class": A 3, B 3, C 2, (empty) 1.'
    unchanged(t, before)


def test_count_each(engine, gappy):
    t = make(gappy, engine)
    before = snapshot(t)
    got = t.count_each("subject")._done()
    if engine == "polars":
        got = got.to_pandas()
    assert list(got.columns) == ["subject", "count", "percent"]
    expected = gappy["subject"].value_counts()
    assert got["subject"].tolist() == expected.index.tolist() == ["maths", "art"]
    assert got["count"].tolist() == expected.tolist() == [5, 4]
    assert got["percent"].tolist() == [55.6, 44.4]
    # empty cells are a value of their own, and ties sort by value
    cls = t.count_each("class")._done()
    if engine == "polars":
        cls = cls.to_pandas()
    assert cls["class"].tolist()[:3] == ["A", "B", "C"] and pd.isna(cls["class"].iloc[3])
    assert cls["count"].tolist() == [3, 3, 2, 1]
    assert cls["percent"].sum() == pytest.approx(100, abs=0.2)
    unchanged(t, before)


def test_count_each_numbers(engine, marks):
    got = make(marks, engine).count_each("test")._done()
    if engine == "polars":
        got = got.to_pandas()
    assert got["test"].tolist() == [1, 2, 3] and got["count"].tolist() == [8, 8, 8]


# ---------------------------------------------------------------- correlation
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_matches_scipy(engine, marks, method, capsys):
    t = make(marks, engine)
    before = snapshot(t)
    r = t.correlation_between("marks", "height", method=method)
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    want_r, want_p = fn(marks["marks"], marks["height"])
    assert isinstance(r, float)
    assert r == pytest.approx(want_r)
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert f"r = {want_r:.2f}" in out and _strength(want_r) in out and _p_words(want_p) in out
    unchanged(t, before)


def test_correlation_skips_empty_rows(engine, gappy, capsys):
    t = make(gappy, engine)
    r = t.correlation_between("hours", "marks")
    both = gappy[["hours", "marks"]].dropna()
    assert r == pytest.approx(stats.pearsonr(both["hours"], both["marks"])[0])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == 'Note: 4 rows with an empty cell in "hours" or "marks" were skipped.'
    assert "5 rows" in out[1]


def test_correlation_words_on_example():
    t = make_table(sdata.example("marks"))
    r = t.correlation_between("hours", "marks")
    ex = sdata.example("marks")
    assert r == pytest.approx(stats.pearsonr(ex["hours"], ex["marks"])[0])


def test_strength_and_p_words():
    assert _strength(0.85) == "very strong, going up"
    assert _strength(-0.65) == "strong, going down"
    assert _strength(0.4) == "medium, going up"
    assert _strength(0.399) == "medium, going up"  # shown as 0.40, so called medium
    assert _strength(-0.25) == "weak, going down"
    assert _strength(0.1) == "very weak, going up"
    assert _strength(0.0) == "no link at all"
    assert _p_words(0.0003) == "very unlikely to be chance (p = 0.0003)"
    assert _p_words(0.00001) == "very unlikely to be chance (p < 0.0001)"
    assert _p_words(0.02) == "unlikely to be chance (p = 0.020)"
    assert _p_words(0.3) == "could easily be chance (p = 0.300)"


# ---------------------------------------------------------------- friendly errors
def test_friendly_errors(engine, gappy):
    t = make(gappy, engine)
    before = snapshot(t)
    with pytest.raises(SimplydataError, match='average_of needs a column of numbers; "name" has text'):
        t.average_of("name")
    with pytest.raises(SimplydataError, match='"passed" has True/False'):
        t.total_of("passed")
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        t.average_of("mark")
    with pytest.raises(SimplydataError, match='Column "klass" not found'):
        t.average_of("marks", for_each="klass")
    with pytest.raises(SimplydataError, match="both"):
        t.average_of("marks", for_each="marks")
    with pytest.raises(SimplydataError, match="in quotes"):
        t.average_of("marks", for_each=3)
    with pytest.raises(SimplydataError, match="in quotes"):
        t.median_of(5)
    with pytest.raises(SimplydataError, match="no option called"):
        t.average_of("marks", for_eech="class")
    with pytest.raises(SimplydataError, match='Did you mean method="spearman"'):
        t.correlation_between("marks", "hours", method="spearmen")
    with pytest.raises(SimplydataError, match="two different columns"):
        t.correlation_between("marks", "marks")
    with pytest.raises(SimplydataError, match='"name" has text'):
        t.correlation_between("marks", "name")
    with pytest.raises(SimplydataError, match="one column name"):
        t.count_each(["class"])
    with pytest.raises(SimplydataError, match="Column"):
        t.count_rows(for_each="klass")
    unchanged(t, before)


def test_correlation_errors(engine):
    t = make(pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [5.0, 5.0, 5.0, 1.0], "c": [1.0, np.nan, 2.0, 3.0]}), engine)
    with pytest.raises(SimplydataError, match='every number in "b" is the same'):
        t.correlation_between("a", "b")
    with pytest.raises(SimplydataError, match="at least 3 rows"):
        t.correlation_between("a", "c")


# ---------------------------------------------------------------- the printed sentence
def test_every_summary_prints_one_sentence(engine, gappy, capsys):
    t = make(gappy, engine)
    t.average_of("marks", for_each="class")
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == 'The average of "marks" for each "class": A 60, B 60, C 90, (empty) 60.'
    t.median_of(["marks", "hours"])
    assert capsys.readouterr().out.splitlines()[-1] == (
        'The median of "marks" is 65 (6 values); the median of "hours" is 2.5 (8 values).')
    t.total_of(["marks", "hours"], for_each="class")
    assert capsys.readouterr().out.splitlines()[-1] == (
        'The total of "marks" and "hours" for each "class": a new table with 4 rows.')


def test_count_each_prints_a_sentence(capsys):
    make_table(sdata.example("fruit")).count_each("shop")
    assert capsys.readouterr().out.startswith('How many rows have each "shop" (2 different values): ')


def test_long_sentences_stop_after_a_few_groups(capsys):
    data = pd.DataFrame({"g": list(range(12)), "x": list(range(12))})
    make_table(data).total_of("x", for_each="g")
    out = capsys.readouterr().out
    assert out.count(",") == GROUPS_IN_SENTENCE - 1 and "and 4 more (the new table has them all)" in out


def test_sentence_numbers_are_short():
    assert _say_number(64.9074) == "64.91"
    assert _say_number(3.0) == "3"
    assert _say_number(0.004213) == "0.0042"
    assert _say_number(float("nan")) == "empty"
