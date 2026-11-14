"""Numbers by position (keep_numbers_by_position, remove_numbers_by_position), replace_numbers and
number_at("last"). Every result is checked against the NumPy expression written next to it."""

import numpy as np
import pytest

import simplydata as sdata
from simplydata import SimplydataError

VALUES = [12, 15, 9, 20, 14, 30, 7, 18, 11, 25]


@pytest.fixture
def nums():
    return sdata.numbers(VALUES)


A = np.array(VALUES)


def test_keep_a_run(nums, capsys):
    nums.keep_numbers_by_position(3, to=8)
    assert "Kept the numbers at positions 3 to 8 (6 of 10 numbers)." in capsys.readouterr().out
    assert nums.as_numpy().tolist() == A[2:8].tolist()  # NumPy a[2:8]


def test_keep_every_second(nums, capsys):
    nums.keep_numbers_by_position(1, step=2)
    assert "every 2nd position from position 1 to position 10: positions 1, 3, 5, 7, 9" in capsys.readouterr().out
    assert nums.as_numpy().tolist() == A[::2].tolist()  # NumPy a[::2]


def test_keep_a_list_and_one(nums):
    nums.keep_numbers_by_position([1, 4])
    assert nums.as_numpy().tolist() == np.take(A, [0, 3]).tolist()  # NumPy np.take(a, [0, 3])
    nums.undo()
    nums.keep_numbers_by_position([4, 1])
    assert nums.as_numpy().tolist() == A[[3, 0]].tolist()
    nums.undo()
    nums.keep_numbers_by_position(5)
    assert nums.as_numpy().tolist() == [A[4]]


def test_keep_to_the_end_and_beyond(nums, capsys):
    nums.keep_numbers_by_position(8, to="last")
    assert nums.as_numpy().tolist() == A[7:].tolist()
    nums.undo()
    capsys.readouterr()
    nums.keep_numbers_by_position(8, to=99)
    assert "There are 10 numbers, so positions 8 to 10 were kept." in capsys.readouterr().out
    assert nums.as_numpy().tolist() == A[7:].tolist()


def test_positions_are_not_values(nums):
    by_position = sdata.numbers(VALUES).keep_numbers_by_position(9, to=12 - 2)
    by_value = sdata.numbers(VALUES).keep_numbers(between=[9, 10])
    assert by_position.as_numpy().tolist() == A[8:10].tolist() == [11, 25]
    assert by_value.as_numpy().tolist() == [9]


def test_remove_numbers_by_position(nums, capsys):
    nums.remove_numbers_by_position("last")
    assert "Removed the number at position 10 (1 of 10 numbers). 9 left." in capsys.readouterr().out
    assert nums.as_numpy().tolist() == np.delete(A, -1).tolist()  # NumPy np.delete(a, -1)
    nums.undo()
    nums.remove_numbers_by_position(2, step=2)
    assert nums.as_numpy().tolist() == np.delete(A, np.s_[1::2]).tolist()
    nums.undo()
    nums.remove_numbers_by_position([1, 3])
    assert nums.as_numpy().tolist() == np.delete(A, [0, 2]).tolist()


def test_undo_and_steps(nums, capsys):
    nums.keep_numbers_by_position(3, to=8)
    nums.remove_numbers_by_position(1)
    nums.steps()
    out = capsys.readouterr().out
    assert "1. keep the numbers at positions 3 to 8" in out and "2. remove the number at position 1" in out
    nums.undo()
    nums.undo()
    assert nums.as_numpy().tolist() == VALUES


def test_empty_numbers_keep_their_place():
    nums = sdata.numbers([1, None, 3])
    nums.keep_numbers_by_position(2, to=3)
    assert np.isnan(nums.as_numpy()[0]) and nums.as_numpy()[1] == 3


@pytest.mark.parametrize("call, message", [
    (lambda n: n.keep_numbers_by_position(), "needs which positions"),
    (lambda n: n.keep_numbers_by_position(0), "Positions count from 1, so the first position is 1, not 0"),
    (lambda n: n.keep_numbers_by_position(-1), 'For the last positions use to="last" or nums.keep_last_numbers(3)'),
    (lambda n: n.keep_numbers_by_position(1.5), "whole numbers from 1 to 10"),
    (lambda n: n.keep_numbers_by_position(True), "whole numbers from 1 to 10"),
    (lambda n: n.keep_numbers_by_position(8, to=3), "Position 8 comes after position 3. Write the smaller first: "
                                                    "nums.keep_numbers_by_position(3, to=8); to turn the order "
                                                    "round use nums.reverse()"),
    (lambda n: n.keep_numbers_by_position([1, 2], step=2), "not with a list"),
    (lambda n: n.keep_numbers_by_position([2, 2]), "Position 2 is listed twice"),
    (lambda n: n.keep_numbers_by_position(11), "There is no position 11"),
    (lambda n: n.keep_numbers_by_position(above=5), "nums.keep_numbers(above=...)"),
    (lambda n: n.remove_numbers_by_position(below=5), "nums.remove_numbers(below=...)"),
    (lambda n: n.keep_numbers_by_position(1, too=3), "Did you mean to=?"),
    (lambda n: n.remove_numbers_by_position(1, to="last"), "That would remove every number"),
])
def test_position_errors(nums, call, message):
    with pytest.raises(SimplydataError) as info:
        call(nums)
    assert message in str(info.value)
    assert not nums._history


def test_no_numbers_to_choose_from():
    with pytest.raises(SimplydataError, match="no numbers to choose from"):
        sdata.numbers([1.0]).remove_empty_numbers().keep_numbers(above=5).keep_numbers_by_position(1)


# ---------------------------------------------------------------- number_at("last")
def test_number_at_last(nums):
    assert nums.number_at("last") == VALUES[-1] == A[-1]
    with pytest.raises(SimplydataError, match='or "last"'):
        nums.number_at("first")


# ---------------------------------------------------------------- replace_numbers
def test_replace_numbers_cap(nums, capsys):
    nums.replace_numbers(above=20, with_value=20)
    assert "Replaced 2 of 10 numbers above 20 with 20." in capsys.readouterr().out
    assert nums.as_numpy().tolist() == np.where(A > 20, 20, A).tolist()  # NumPy np.where(a > 20, 20, a)
    nums.undo()
    assert nums.as_numpy().tolist() == VALUES


def test_replace_a_missing_value_code():
    nums = sdata.numbers([3, -999, 5, -999])
    nums.replace_numbers(-999, with_value=None)
    a = np.array([3, -999, 5, -999], dtype=float)
    a[a == -999] = np.nan  # NumPy a[a == -999] = np.nan; pandas s.replace(-999, np.nan)
    np.testing.assert_array_equal(nums.as_numpy(), a)
    assert nums.count_empty_numbers() == 2


def test_replace_a_list_of_old_numbers():
    nums = sdata.numbers([3, -99, 5, -999])
    nums.replace_numbers([-99, -999], with_value=0)
    assert nums.as_numpy().tolist() == [3, 0, 5, 0]


@pytest.mark.parametrize("option, value, native", [
    ("above", 20, lambda a: a > 20), ("below", 10, lambda a: a < 10), ("at_least", 20, lambda a: a >= 20),
    ("at_most", 9, lambda a: a <= 9), ("equal_to", 15, lambda a: a == 15), ("not_equal_to", 15, lambda a: a != 15),
    ("between", [10, 15], lambda a: (a >= 10) & (a <= 15)), ("one_of", [7, 30], lambda a: np.isin(a, [7, 30])),
])
def test_replace_numbers_each_condition(nums, option, value, native):
    nums.replace_numbers(with_value=-1, **{option: value})
    assert nums.as_numpy().tolist() == np.where(native(A), -1, A).tolist()


def test_empty_numbers_never_match_but_is_empty_does():
    nums = sdata.numbers([1, None, 30])
    nums.replace_numbers(not_equal_to=1, with_value=0)
    out = nums.as_numpy()
    assert out[0] == 1 and np.isnan(out[1]) and out[2] == 0
    nums.replace_numbers(is_empty=True, with_value=5)
    assert nums.as_numpy().tolist() == [1, 5, 0]


@pytest.mark.parametrize("call, message", [
    (lambda n: n.replace_numbers(), "needs the number to replace, or one condition"),
    (lambda n: n.replace_numbers(above=5), "needs with_value="),
    (lambda n: n.replace_numbers(5, above=2, with_value=1), "either the old number or one condition"),
    (lambda n: n.replace_numbers(above=5, below=2, with_value=1), "one condition at a time"),
    (lambda n: n.replace_numbers(above=5, with_value="x"), "with_value= needs a number, or None"),
    (lambda n: n.replace_numbers("5", with_value=1), "The old value needs to be a number"),
    (lambda n: n.replace_numbers(None, with_value=1), "replace_numbers(is_empty=True, with_value=0)"),
    (lambda n: n.replace_numbers(abuve=5, with_value=1), "Did you mean above=?"),
])
def test_replace_numbers_errors(nums, call, message):
    with pytest.raises(SimplydataError) as info:
        call(nums)
    assert message in str(info.value)
    assert not nums._history


@pytest.mark.parametrize("name, word", [
    ("take", "keep_numbers_by_position"), ("slice", "keep_numbers_by_position"),
    ("keep_numbers_at", "keep_numbers_by_position"), ("keep_positions", "keep_numbers_by_position"),
    ("replace", "replace_numbers"), ("replace_values", "replace_numbers"), ("where", "replace_numbers"),
    ("clip", "replace_numbers"), ("putmask", "replace_numbers"),
])
def test_other_names_point_to_the_new_words(nums, name, word):
    with pytest.raises(AttributeError, match=rf"nums\.{word}\(\)"):
        getattr(nums, name)


def test_brackets_mention_the_new_words(nums):
    with pytest.raises(SimplydataError) as info:
        nums[2:5]
    assert "nums.keep_numbers_by_position(3, to=8)" in str(info.value)
    assert 'nums.number_at("last")' in str(info.value)
