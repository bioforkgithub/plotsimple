"""Numbers and number grids: every word checked against NumPy."""

import numpy as np
import pandas as pd
import pytest

import plotsimple as ps
from plotsimple import PlotsimpleError
from plotsimple._data import frame_for, is_listlike, load
from plotsimple.numbers import NumberGrid, Numbers, number_grid, numbers, random_numbers

NAN = np.nan


def same(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


# ====================================================================== making numbers
def test_from_list_and_from_several_numbers():
    assert isinstance(numbers([3, 5, 2, 8]), Numbers)
    same(numbers([3, 5, 2, 8]), [3, 5, 2, 8])
    same(numbers(3, 5, 2, 8), [3, 5, 2, 8])
    assert numbers([3, 5]).done().dtype.kind == "i"  # whole numbers stay whole
    same(numbers([1, None, 3]), [1, NAN, 3])
    same(numbers(np.array([1.5, 2.5])), [1.5, 2.5])
    same(numbers(pd.Series([1, 2])), [1, 2])
    same(numbers(range(3)), [0, 1, 2])
    assert numbers([]).count() == 0


def test_counting_includes_the_end():
    same(numbers(1, to=10), np.arange(1, 11))
    same(numbers(0, to=100, step=25), [0, 25, 50, 75, 100])
    same(numbers(10, to=1), np.arange(10, 0, -1))
    same(numbers(0, to=1, step=0.1), np.linspace(0, 1, 11))
    same(numbers(0, to=1, how_many=5), np.linspace(0, 1, 5))
    same(numbers(1, to=10, step=4), [1, 5, 9])
    assert numbers(1, to=10).done().dtype.kind == "i"


def test_making_numbers_errors():
    with pytest.raises(PlotsimpleError, match="ps.numbers\\(1, to=5\\)"):
        numbers(5)
    with pytest.raises(PlotsimpleError, match="text"):
        numbers(["a", 2])
    with pytest.raises(PlotsimpleError, match="without quotes"):
        numbers(["3", "4"])
    with pytest.raises(PlotsimpleError, match="True/False"):
        numbers([True, False])
    with pytest.raises(PlotsimpleError, match="number_grid"):
        numbers([[1, 2], [3, 4]])
    with pytest.raises(PlotsimpleError, match="never gets there"):
        numbers(1, to=10, step=-1)
    with pytest.raises(PlotsimpleError, match="not both"):
        numbers(0, to=1, step=0.1, how_many=3)
    with pytest.raises(PlotsimpleError, match="not 0"):
        numbers(0, to=1, step=0)
    with pytest.raises(PlotsimpleError, match="has no option called start="):
        numbers(start=1, to=10)


def test_random_numbers_match_numpy():
    same(random_numbers(how_many=5, seed=1), np.random.default_rng(1).uniform(0, 1, 5))
    same(random_numbers(how_many=5, between=[10, 2], seed=2), np.random.default_rng(2).uniform(2, 10, 5))
    dice = random_numbers(how_many=500, between=[1, 6], whole_numbers=True, seed=3)
    same(dice, np.random.default_rng(3).integers(1, 6, 500, endpoint=True))
    assert dice.smallest() == 1 and dice.biggest() == 6
    same(random_numbers(how_many=4, average=50, standard_deviation=10, seed=4),
         np.random.default_rng(4).normal(50, 10, 4))
    grid = random_numbers(rows=2, columns=3, seed=5)
    assert isinstance(grid, NumberGrid) and grid.shape() == (2, 3)
    same(random_numbers(how_many=3, seed=9), random_numbers(how_many=3, seed=9))


def test_random_numbers_errors():
    with pytest.raises(PlotsimpleError, match="how_many"):
        random_numbers(seed=1)
    with pytest.raises(PlotsimpleError, match="not both"):
        random_numbers(how_many=3, between=[0, 1], average=2, standard_deviation=1)
    with pytest.raises(PlotsimpleError, match="seed="):
        random_numbers(how_many=3, seed=-1)
    with pytest.raises(PlotsimpleError, match="both average= and standard_deviation="):
        random_numbers(how_many=3, average=5)
    with pytest.raises(PlotsimpleError, match="whole numbers in between"):
        random_numbers(how_many=3, between=[0.5, 2], whole_numbers=True)


def test_making_grids():
    grid = number_grid([[1, 2, 3], [4, 5, 6]])
    assert isinstance(grid, NumberGrid)
    assert grid.shape() == (2, 3) and grid.count_rows() == 2 and grid.count_columns() == 3
    assert grid.count() == 6 and len(grid) == 2
    same(number_grid(rows=2, columns=3), np.zeros((2, 3)))
    same(number_grid(rows=2, columns=2, filled_with=1), np.ones((2, 2)))
    same(number_grid(rows=1, columns=2, filled_with=None), [[NAN, NAN]])
    same(number_grid([1, 2, 3, 4, 5, 6], columns=3), np.arange(1, 7).reshape(2, 3))
    same(number_grid([1, 2, 3, 4, 5, 6], rows=3), np.arange(1, 7).reshape(3, 2))
    same(numbers(1, to=6).number_grid(columns=2), np.arange(1, 7).reshape(3, 2))
    with pytest.raises(PlotsimpleError, match="same length"):
        number_grid([[1, 2], [3]])
    with pytest.raises(PlotsimpleError, match="cannot fill"):
        number_grid([1, 2, 3, 4, 5], columns=2)
    with pytest.raises(PlotsimpleError, match="rows= and columns="):
        number_grid()


# ====================================================================== looking and picking
def test_positions_count_from_one():
    nums = numbers([3, 5, 2, 8])
    assert nums.number_at(1) == 3 and nums.number_at(4) == 8
    assert isinstance(nums.number_at(1), int)
    with pytest.raises(PlotsimpleError, match="from 1 to 4"):
        nums.number_at(0)
    with pytest.raises(PlotsimpleError, match="count from 1"):
        nums[0]
    grid = number_grid([[1, 2, 3], [4, 5, 6]])
    assert grid.number_at(row=2, column=3) == 6
    same(grid.row(1), [1, 2, 3])
    same(grid.column(2), [2, 5])
    same(grid.numbers(), [1, 2, 3, 4, 5, 6])
    with pytest.raises(PlotsimpleError, match="from 1 to 2"):
        grid.row(3)
    with pytest.raises(PlotsimpleError, match="number_at"):
        grid[0, 0]


def test_show_words(capsys):
    nums = numbers(1, to=10)
    nums.show_first(3)
    nums.show_last(2)
    nums.show()
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[1, 2, 3]" and out[1] == "[ 9, 10]"
    assert "plotsimple numbers: 10 numbers" in repr(nums)
    assert "2 rows, 3 columns" in repr(number_grid(rows=2, columns=3))


def test_keep_and_remove_use_table_conditions(capsys):
    values = np.array([3, 50, 70, NAN, 51, 40])
    nums = numbers(values)
    nums.keep_numbers(above=50)
    same(nums, [70, 51])
    assert "Kept 2 of 6 numbers." in capsys.readouterr().out
    nums.undo()
    nums.remove_numbers(between=[40, 60])
    same(nums, [3, 70, NAN])
    assert "Removed 3 of 6 numbers. 3 left." in capsys.readouterr().out
    for option, expected in [
        (dict(below=40), [3]), (dict(at_least=50), [50, 70, 51]), (dict(at_most=40), [3, 40]),
        (dict(equal_to=70), [70]), (dict(not_equal_to=70), [3, 50, 51, 40]),
        (dict(one_of=[3, 40]), [3, 40]), (dict(is_empty=True), [NAN]),
        (dict(is_empty=False), [3, 50, 70, 51, 40]), (dict(between=[60, 40]), [50, 51, 40]),
    ]:
        nums = numbers(values)
        nums.keep_numbers(**option)
        same(nums, expected)


def test_keep_errors():
    nums = numbers([1, 2, 3])
    with pytest.raises(PlotsimpleError, match="one condition"):
        nums.keep_numbers()
    with pytest.raises(PlotsimpleError, match="one condition at a time"):
        nums.keep_numbers(above=1, below=3)
    with pytest.raises(PlotsimpleError, match="without quotes"):
        nums.keep_numbers(above="2")
    with pytest.raises(PlotsimpleError, match="two numbers"):
        nums.keep_numbers(between=5)
    with pytest.raises(PlotsimpleError, match="no option called contains="):
        nums.keep_numbers(contains="1")
    with pytest.raises(PlotsimpleError, match="above=50"):
        nums.keep_numbers(50)


def test_first_last_random_empty_duplicate():
    nums = numbers([4, 1, 4, NAN, 2, 1])
    nums.keep_first_numbers(3)
    same(nums, [4, 1, 4])
    nums.start_again()
    nums.keep_last_numbers(2)
    same(nums, [2, 1])
    nums.start_again()
    nums.remove_empty_numbers()
    same(nums, [4, 1, 4, 2, 1])
    nums.remove_duplicate_numbers()
    same(nums, [4, 1, 2])
    picked = numbers(1, to=20)
    picked.keep_random_numbers(5, seed=1)
    expected = np.sort(np.random.default_rng(1).choice(20, 5, replace=False)) + 1
    same(picked, expected)


def test_fill_empty(capsys):
    values = [1, NAN, 3, NAN]
    for kw, expected in [
        (dict(with_value=0), [1, 0, 3, 0]), (dict(with_the="average"), [1, 2, 3, 2]),
        (dict(with_the="median"), [1, 2, 3, 2]), (dict(with_the="value above"), [1, 1, 3, 3]),
        (dict(with_the="value below"), [1, 3, 3, NAN]),
    ]:
        nums = numbers(values)
        nums.fill_empty(**kw)
        same(nums, expected)
    assert "1 value stayed empty" in capsys.readouterr().out
    with pytest.raises(PlotsimpleError, match="exactly one"):
        numbers(values).fill_empty()
    with pytest.raises(PlotsimpleError, match="with_the"):
        numbers(values).fill_empty(with_the="mode")


# ====================================================================== maths words and symbols
@pytest.mark.parametrize("word, symbol, other", [
    ("add", lambda a, b: a + b, 5), ("subtract", lambda a, b: a - b, 2),
    ("times", lambda a, b: a * b, 3), ("divide_by", lambda a, b: a / b, 4),
    ("power", lambda a, b: a ** b, 2), ("add", lambda a, b: a + b, [1, 2, 3, 4]),
    ("times", lambda a, b: a * b, [0.5, 1, 2, 3]),
])
def test_words_change_and_symbols_make_new(word, symbol, other):
    base = np.array([3.0, 5.0, 2.0, 8.0])
    nums = numbers(base)
    new = symbol(nums, other)
    assert isinstance(new, Numbers)
    same(nums, base)  # symbols leave the numbers alone
    same(new, symbol(base, np.asarray(other)))
    getattr(nums, word)(other)
    same(nums, new)  # the word gives the same answer
    nums.undo()
    same(nums, base)


def test_reflected_symbols_and_numbers_with_numbers():
    nums = numbers([1, 2, 4])
    same(10 - nums, [9, 8, 6])
    same(8 / nums, [8, 4, 2])
    same(2 ** nums, [2, 4, 16])
    same([1, 1, 1] + nums, [2, 3, 5])
    same(nums + numbers([1, 1, 1]), [2, 3, 5])
    same(-nums, [-1, -2, -4])
    same(abs(numbers([-1, 2])), [1, 2])
    same(np.array([1, 1, 1]) * nums, [1, 2, 4])
    assert isinstance(np.float64(2) * nums, Numbers)
    with pytest.raises(PlotsimpleError, match="3 numbers"):
        nums + [1, 2]
    with pytest.raises(PlotsimpleError, match="without quotes"):
        nums.add("5")
    with pytest.raises(PlotsimpleError, match="v0.7"):
        nums @ nums


def test_division_by_zero_becomes_empty(capsys):
    nums = numbers([4, 0, NAN])
    same(nums / 0, [NAN, NAN, NAN])
    assert "dividing by 0 has no answer, so 2 numbers are now empty" in capsys.readouterr().out
    nums.divide_by([2, 0, 1])
    same(nums, [2, NAN, NAN])
    assert "1 number is now empty" in capsys.readouterr().out
    same(1 / numbers([0, 2]), [NAN, 0.5])


def test_power_edge_cases(capsys):
    same(numbers([2, 3]) ** -1, [0.5, 1 / 3])
    same(numbers([0, 4]) ** -1, [NAN, 0.25])
    assert "minus power" in capsys.readouterr().out
    same(numbers([-4, 4]) ** 0.5, [NAN, 2])
    assert "power with decimals" in capsys.readouterr().out
    big = numbers([10]) ** 30
    assert big.number_at(1) == pytest.approx(1e30)  # no silent whole-number overflow
    assert (numbers([2, 3]) ** 2).done().dtype.kind == "i"


def test_one_at_a_time_words(capsys):
    v = np.array([1.0, 4.0, -9.0, 100.0, 0.0])
    checks = [
        ("square_root", {}, np.where(v < 0, NAN, np.sqrt(np.abs(v)))),
        ("absolute_value", {}, np.abs(v)),
        ("log", {}, np.where(v <= 0, NAN, np.log(np.abs(v) + (v == 0)))),
        ("log", {"base": 10}, np.where(v <= 0, NAN, np.log10(np.abs(v) + (v == 0)))),
        ("log", {"base": 2}, np.where(v <= 0, NAN, np.log2(np.abs(v) + (v == 0)))),
        ("log", {"base": 5}, np.where(v <= 0, NAN, np.log(np.abs(v) + (v == 0)) / np.log(5))),
        ("exp", {}, np.exp(v)),
    ]
    for word, kw, expected in checks:
        nums = numbers(v)
        getattr(nums, word)(**kw)
        same(nums, expected)
    out = capsys.readouterr().out
    assert "a negative number has no square root, so 1 number is now empty" in out
    assert "the log only exists for numbers above 0, so 2 numbers are now empty" in out
    with pytest.raises(PlotsimpleError, match="base="):
        numbers(v).log(base=1)


def test_round_half_to_even():
    nums = numbers([0.5, 1.5, 2.5, 3.5, 2.675, 1234])
    nums.round_to(0)
    same(nums, np.round([0.5, 1.5, 2.5, 3.5, 2.675, 1234]))
    same(nums.done()[:4], [0, 2, 2, 4])
    nums.undo()
    nums.round_to(2)
    same(nums, np.round([0.5, 1.5, 2.5, 3.5, 2.675, 1234], 2))
    nums.round_to(-2)
    assert nums.number_at(6) == 1200
    with pytest.raises(PlotsimpleError, match="whole number"):
        nums.round_to(1.5)


def test_order_words():
    nums = numbers([3, NAN, 1, 2])
    nums.sort()
    same(nums, [1, 2, 3, NAN])
    nums.sort(biggest_first=True)
    same(nums, [3, 2, 1, NAN])
    nums.reverse()
    same(nums, [NAN, 1, 2, 3])
    nums = numbers([1, 2, NAN, 3])
    nums.running_total()
    same(nums, pd.Series([1, 2, NAN, 3]).cumsum())
    with pytest.raises(PlotsimpleError, match="True or False"):
        nums.sort(biggest_first="yes")


def test_history_words(capsys):
    nums = numbers([1, 2, 3])
    nums.add(1)
    nums.times(2)
    nums.steps()
    assert "1. add 1\n2. times 2" in capsys.readouterr().out
    nums.start_again()
    same(nums, [1, 2, 3])
    nums.undo()
    same(nums, [4, 6, 8])
    nums.undo()
    nums.undo()
    nums.undo()
    assert "Nothing to undo" in capsys.readouterr().out
    same(nums, [1, 2, 3])


# ====================================================================== summaries
def test_summaries_match_numpy(capsys):
    v = np.array([3.0, 5.0, NAN, 2.0, 8.0])
    nums = numbers(v)
    assert nums.average() == pytest.approx(np.nanmean(v))
    assert nums.median() == pytest.approx(np.nanmedian(v))
    assert nums.total() == pytest.approx(np.nansum(v))
    assert nums.smallest() == 2 and nums.biggest() == 8
    assert nums.standard_deviation() == pytest.approx(np.nanstd(v, ddof=1))
    assert nums.standard_deviation() == pytest.approx(pd.Series(v).std())
    assert nums.percentile(90) == pytest.approx(np.nanpercentile(v, 90))
    assert nums.count() == 5 and nums.count_empty() == 1 and len(nums) == 5
    assert "1 empty value was skipped" in capsys.readouterr().out
    assert isinstance(numbers([1, 2]).total(), int)


def test_summaries_on_nothing(capsys):
    empty = numbers([NAN, NAN])
    assert np.isnan(empty.average())
    assert empty.total() == 0
    assert np.isnan(numbers([4]).standard_deviation())
    assert "at least 2 numbers" in capsys.readouterr().out
    with pytest.raises(PlotsimpleError, match="0 to 100"):
        numbers([1, 2]).percentile(120)


def test_count_each():
    table = numbers([2, 1, 2, NAN, 2, 1]).count_each()
    plain = table.done()
    assert list(plain.columns) == ["number", "count", "percent"]
    assert plain["number"].tolist()[:2] == [2, 1]
    assert plain["count"].tolist() == [3, 2, 1]
    assert plain["percent"].tolist() == [50.0, 33.3, 16.7]


def test_summary_names_and_hints():
    nums = numbers([1, 2])
    with pytest.raises(PlotsimpleError, match=r"nums\.average\(\)"):
        nums.average_of()
    with pytest.raises(PlotsimpleError, match=r"nums\.average\(\)"):
        nums.mean()
    with pytest.raises(PlotsimpleError, match=r"keep_numbers"):
        nums.keep_rows()
    with pytest.raises(PlotsimpleError, match="brackets"):
        nums.total = 3
    assert not hasattr(nums, "dtype")  # still an AttributeError for Python tools


# ====================================================================== grids
def test_grid_summaries_for_each():
    g = np.array([[1.0, 2.0, NAN], [4.0, 5.0, 6.0]])
    grid = number_grid(g)
    assert grid.average() == pytest.approx(np.nanmean(g))
    same(grid.average(for_each="row"), np.nanmean(g, axis=1))
    same(grid.total(for_each="column"), np.nansum(g, axis=0))
    same(grid.biggest(for_each="row"), np.nanmax(g, axis=1))
    same(grid.standard_deviation(for_each="column"), [np.std([1, 4], ddof=1), np.std([2, 5], ddof=1), NAN])
    same(grid.percentile(50, for_each="row"), np.nanmedian(g, axis=1))
    assert isinstance(grid.median(for_each="row"), Numbers)
    with pytest.raises(PlotsimpleError, match='Did you mean for_each="row"'):
        grid.average(for_each="rows")


def test_grid_maths_and_swap():
    g = np.arange(1, 7).reshape(2, 3)
    grid = number_grid(g)
    same(grid * 2, g * 2)
    same(grid + grid, g + g)
    same(grid + [10, 20, 30], g + [10, 20, 30])  # one number per column
    same(numbers([1, 1, 1]) + grid, g + 1)
    assert isinstance(grid ** 2, NumberGrid)
    grid.square_root()
    same(grid, np.sqrt(g))
    grid.undo()
    grid.swap_rows_and_columns()
    same(grid, g.T)
    assert grid.shape() == (3, 2)
    grid.undo()
    same(grid, g)
    with pytest.raises(PlotsimpleError, match="one for each column"):
        grid + [1, 2]
    same(np.sin(grid), np.sin(g))
    assert list(grid) == [[1, 2, 3], [4, 5, 6]]


# ====================================================================== finishing and charts
def test_done_table_save(tmp_path):
    nums = numbers([1, 2, 3])
    plain = nums.done()
    assert isinstance(plain, np.ndarray)
    plain[0] = 99
    assert nums.number_at(1) == 1  # done() gives a copy
    assert nums.table(name="marks").column_names() == ["marks"]
    nums.save(tmp_path / "nums.csv")
    assert pd.read_csv(tmp_path / "nums.csv")["value"].tolist() == [1, 2, 3]
    grid = number_grid([[1, 2], [3, 4]])
    assert grid.table().column_names() == ["1", "2"]
    grid.save(tmp_path / "grid.csv")
    assert pd.read_csv(tmp_path / "grid.csv").values.tolist() == [[1, 2], [3, 4]]
    with pytest.raises(PlotsimpleError, match="file type"):
        nums.save(tmp_path / "nums")
    same(np.asarray(nums), [1, 2, 3])


def test_charts_and_tables_accept_numbers():
    nums = numbers([3, 5, 2])
    assert is_listlike(nums)
    assert load(nums)["value"].tolist() == [3, 5, 2]
    grid = number_grid([[1, 2], [3, 4]])
    assert not is_listlike(grid)
    frame = load(grid)
    assert list(frame.columns) == ["1", "2"] and list(frame.index) == [1, 2]
    table, names = frame_for(pd.DataFrame(), y=nums)
    assert table[names["y"]].tolist() == [3, 5, 2]
    ps.chart().lines(nums).dots(x=nums, y=nums * 2).histogram(x=nums)._render()
    ps.chart(grid).heatmap(numbers=True)._render()
    grid.chart().heatmap()._render()
    nums.chart().bars(x=numbers(1, to=3), y="value")._render()
    assert ps.table(nums).column_names() == ["value"]
    assert ps.table(grid).count_rows() == 2


def test_every_word_has_a_docstring_with_an_example():
    # words shared by lists and grids (add, log, undo ...) show the list example
    for cls, name in ((Numbers, "nums."), (NumberGrid, "grid.")):
        for word in cls._word_names():
            doc = getattr(cls, word).__doc__ or ""
            assert doc.strip(), word
            assert name in doc or "nums." in doc, f"{cls.__name__}.{word} has no example"
