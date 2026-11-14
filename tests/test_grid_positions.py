"""Parts of a grid by position, grid.replace_numbers, "last" in row/column/number_at, and the
pointer for grid.keep_numbers. Every result is checked against the NumPy expression next to it."""

import numpy as np
import pytest

import simplydata as sdata
from simplydata import SimplydataError

G = np.arange(1, 25).reshape(4, 6) * 10 - 100  # 4 x 6, some below 0


@pytest.fixture
def grid():
    return sdata.number_grid(G.tolist())


def test_a_block_is_two_steps(grid, capsys):
    grid.keep_rows_by_position(2, to=3).keep_columns_by_position(1, to=3)
    out = capsys.readouterr().out
    assert "Kept rows 2 to 3 (2 of 4 rows); the grid is now 2 rows by 6 columns." in out
    assert "Kept columns 1 to 3 (3 of 6 columns); the grid is now 2 rows by 3 columns." in out
    np.testing.assert_array_equal(grid.as_numpy(), G[1:3, 0:3])  # NumPy g[1:3, 0:3]
    grid.undo()
    np.testing.assert_array_equal(grid.as_numpy(), G[1:3])
    grid.undo()
    np.testing.assert_array_equal(grid.as_numpy(), G)


def test_every_nth_row_and_lists(grid):
    grid.keep_rows_by_position(1, step=2)
    np.testing.assert_array_equal(grid.as_numpy(), G[::2])  # NumPy g[::2]
    grid.undo()
    grid.keep_columns_by_position([1, 3])
    np.testing.assert_array_equal(grid.as_numpy(), G[:, [0, 2]])  # NumPy g[:, [0, 2]]
    grid.undo()
    grid.keep_rows_by_position([1, 3]).keep_columns_by_position([2, 4])
    np.testing.assert_array_equal(grid.as_numpy(), G[np.ix_([0, 2], [1, 3])])  # NumPy np.ix_
    grid.undo()
    grid.undo()
    grid.keep_columns_by_position(5, to="last")
    np.testing.assert_array_equal(grid.as_numpy(), G[:, 4:])


def test_remove_rows_and_columns(grid, capsys):
    grid.remove_columns_by_position(1)
    assert "Removed column 1 (1 of 6 columns); the grid is now 4 rows by 5 columns." in capsys.readouterr().out
    np.testing.assert_array_equal(grid.as_numpy(), np.delete(G, 0, axis=1))  # NumPy np.delete(g, 0, axis=1)
    grid.undo()
    grid.remove_rows_by_position("last")
    np.testing.assert_array_equal(grid.as_numpy(), np.delete(G, -1, axis=0))
    grid.undo()
    grid.remove_rows_by_position(1, step=2)
    np.testing.assert_array_equal(grid.as_numpy(), np.delete(G, np.s_[::2], axis=0))


def test_one_row_left_is_still_a_grid(grid, capsys):
    grid.keep_rows_by_position(2)
    out = capsys.readouterr().out
    assert "One row is left, so this is still a grid; grid.row(1) gives it as numbers." in out
    assert grid.shape() == (1, 6)
    np.testing.assert_array_equal(grid.row(1).as_numpy(), G[1])


def test_to_beyond_the_end(grid, capsys):
    grid.keep_rows_by_position(3, to=10)
    assert "The grid has 4 rows, so rows 3 to 4 were kept." in capsys.readouterr().out
    np.testing.assert_array_equal(grid.as_numpy(), G[2:])


def test_steps_list_each_part(grid, capsys):
    grid.keep_rows_by_position(2, to=3).keep_columns_by_position(1, to=3)
    grid.steps()
    out = capsys.readouterr().out
    assert "1. keep rows 2 to 3" in out and "2. keep columns 1 to 3" in out


@pytest.mark.parametrize("call, message", [
    (lambda g: g.keep_rows_by_position(0), "the first row is 1, not 0"),
    (lambda g: g.keep_columns_by_position(7), "There is no column 7"),
    (lambda g: g.keep_rows_by_position(3, to=1), "Row 3 comes after row 1. Write the smaller first: "
                                                 "grid.keep_rows_by_position(1, to=3)"),
    (lambda g: g.keep_rows_by_position(-1), '"last" on its own is the last row'),
    (lambda g: g.remove_rows_by_position(1, to="last"), "That would remove every row"),
    (lambda g: g.remove_columns_by_position([1, 2, 3, 4, 5, 6]), "That would remove every column"),
    (lambda g: g.keep_rows_by_position(above=5), "A grid keeps its rows and columns"),
    (lambda g: g.keep_rows_by_position(1, stop=2), "Did you mean step="),
])
def test_grid_position_errors(grid, call, message):
    with pytest.raises(SimplydataError) as info:
        call(grid)
    assert message in str(info.value)
    assert not grid._history


def test_last_in_row_column_and_number_at(grid):
    np.testing.assert_array_equal(grid.row("last").as_numpy(), G[-1])
    np.testing.assert_array_equal(grid.column("last").as_numpy(), G[:, -1])
    assert grid.number_at(row="last", column=1) == G[-1, 0]
    assert grid.number_at(row="last", column="last") == G[-1, -1]
    with pytest.raises(SimplydataError, match='or "last"'):
        grid.row(9)


def test_grid_replace_numbers_keeps_the_shape(grid, capsys):
    grid.replace_numbers(below=0, with_value=0)
    n = int((G < 0).sum())
    assert f"Replaced {n} of 24 numbers below 0 with 0." in capsys.readouterr().out
    np.testing.assert_array_equal(grid.as_numpy(), np.where(G < 0, 0, G))  # NumPy np.where(g < 0, 0, g)
    assert grid.shape() == (4, 6)
    grid.undo()
    np.testing.assert_array_equal(grid.as_numpy(), G)


def test_grid_replace_empty_numbers():
    grid = sdata.number_grid([[1, None], [None, 4]])
    grid.replace_numbers(is_empty=True, with_value=0)
    np.testing.assert_array_equal(grid.as_numpy(), [[1, 0], [0, 4]])
    grid.replace_numbers(above=3, with_value=None)
    assert np.isnan(grid.as_numpy()[1, 1])


@pytest.mark.parametrize("name", ["keep_numbers", "remove_numbers"])
def test_grid_keep_numbers_explains_why_not(grid, name):
    with pytest.raises(AttributeError) as info:
        getattr(grid, name)
    text = str(info.value)
    assert "A grid keeps its rows and columns, so it cannot drop single numbers" in text
    assert "grid.numbers().keep_numbers(above=5)" in text and "grid.replace_numbers(above=5, with_value=0)" in text


@pytest.mark.parametrize("name, word", [
    ("keep_rows", "keep_rows_by_position"), ("keep_columns", "keep_columns_by_position"),
    ("remove_rows", "remove_rows_by_position"), ("remove_columns", "remove_columns_by_position"),
    ("where", "replace_numbers"), ("clip", "replace_numbers"), ("delete", "remove_rows_by_position"),
])
def test_grid_other_names(grid, name, word):
    with pytest.raises(AttributeError, match=rf"grid\.{word}\(\)"):
        getattr(grid, name)


def test_grid_table_rows_are_numbered_from_1(grid, capsys):
    grid.table().show()
    lines = [line for line in capsys.readouterr().out.splitlines() if line[:1].isdigit()]
    assert [int(line.split()[0]) for line in lines] == [1, 2, 3, 4]


def test_grid_brackets_mention_a_block(grid):
    with pytest.raises(SimplydataError, match=r"keep_rows_by_position\(2, to=5\)\.keep_columns_by_position"):
        grid[1:5, 0:3]
