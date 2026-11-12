"""Words for a list of numbers: picking, keeping, ordering and filling.

The names follow tables, with "numbers" in place of "rows", so they can be guessed:

    table.keep_rows("marks", above=50)    ->  nums.keep_numbers(above=50)
    table.keep_first_rows(10)             ->  nums.keep_first_numbers(10)
    table.show_first_rows(5)              ->  nums.show_first_numbers(5)
    table.remove_rows_with_empty_cells()  ->  nums.remove_empty_numbers()
    table.count_empty_cells()             ->  nums.count_empty_numbers()
    table.fill_empty_cells("marks", with_the="average")  ->  nums.fill_empty_numbers(with_the="average")

Conditions are the same as keep_rows (without the text ones), and empty values never
match a condition except is_empty=True.

Positions count from 1: nums.number_at(1) is the first number.
"""

from __future__ import annotations

import sys

import numpy as np

from .._errors import SimplydataError, choice_error, friendly
from ._base import NOT_GIVEN, amount, check_seed, check_whole, is_number, plain, say

CONDITIONS = ("above", "below", "at_least", "at_most", "equal_to", "not_equal_to", "between", "one_of",
              "is_empty")
FILL_CHOICES = ("average", "median", "value above", "value below", "line between neighbours")


class PickingWords:
    # ================================================================== helpers
    def _condition(self, step: str, options: dict, extra: tuple):
        """Which numbers match one condition. Gives (True/False for each number, words)."""
        example = f"{step}(above=50)"
        if extra:
            raise SimplydataError(
                f".{step}() needs one plain word saying which numbers, like above=50 or between=[40, 60]."
                + self._example(example)
            )
        given = {k: v for k, v in options.items() if v is not NOT_GIVEN}
        if not given:
            raise SimplydataError(
                f".{step}() needs one condition saying which numbers. You can use: "
                + ", ".join(f"{c}=" for c in CONDITIONS) + self._example(example)
            )
        if len(given) > 1:
            names = " and ".join(f"{c}=" for c in given)
            raise SimplydataError(
                f".{step}() takes one condition at a time, but you gave {names}. "
                "Write one step per condition, or use between=[low, high]."
                + self._example(f"{step}(at_least=40)   then   nums.{step}(at_most=80)")
            )
        option, value = next(iter(given.items()))
        v = self._values
        empty = self._empty_mask()
        if option == "is_empty":
            if not isinstance(value, bool):
                raise SimplydataError("is_empty= needs True or False (no quotes)." + self._example(f"{step}(is_empty=True)"))
            return (empty if value else ~empty), ("are empty" if value else "are not empty")

        def number(x, example_code=example):
            if is_number(x):
                return x
            hint = ""
            if isinstance(x, str):
                try:
                    float(x)
                    hint = f" Write the number without quotes: {option}={x.strip()}."
                except ValueError:
                    pass
            raise SimplydataError(f"{option}= needs a number (you gave {x!r}).{hint}" + self._example(example_code))

        with np.errstate(invalid="ignore"):
            if option in ("above", "below", "at_least", "at_most"):
                x = number(value)
                cond = {"above": v > x, "below": v < x, "at_least": v >= x, "at_most": v <= x}[option]
                words = f"{option.replace('_', ' ')} {say(x)}"
            elif option == "between":
                ex = f"{step}(between=[40, 60])"
                if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise SimplydataError(
                        "between= needs two numbers in square brackets: the lowest and the highest."
                        + self._example(ex)
                    )
                low, high = sorted(number(x, ex) for x in value)
                cond, words = (v >= low) & (v <= high), f"between {say(low)} and {say(high)}"
            elif option == "one_of":
                ex = f"{step}(one_of=[1, 2, 3])"
                if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)) or not value:
                    raise SimplydataError(
                        "one_of= needs a list of numbers in square brackets. For a single number use equal_to="
                        + self._example(ex)
                    )
                values = [number(x, ex) for x in value]
                cond, words = np.isin(v, values), "one of " + ", ".join(say(x) for x in values)
            else:  # equal_to, not_equal_to
                x = number(value, f"{step}(equal_to=5)")
                if option == "equal_to":
                    cond, words = v == x, f"equal to {say(x)}"
                else:
                    cond, words = v != x, f"not equal to {say(x)}"
        return cond & ~empty, words

    def _keep(self, what: str, keep_mask, kept_word: bool = True):
        before = self._values.size
        new = self._values[keep_mask]
        if kept_word:
            print(f"Kept {new.size} of {amount(before)}.")
        else:
            print(f"Removed {before - new.size} of {amount(before)}. {new.size} left.")
        return self._change(what, new)

    def _check_position(self, step, position, option="position"):
        example = f"{step}(1)" if option == "position" else f"{step}(row=1, column=2)"
        n = self._values.size if option == "position" else self._values.shape[0 if option == "row" else 1]
        if not isinstance(position, (int, np.integer)) or isinstance(position, bool) or not 1 <= position <= n:
            if n == 0:
                raise SimplydataError(f"There are no {option}s to pick from: these {self._noun} are empty.")
            raise SimplydataError(
                f"{option} needs a whole number from 1 to {n}: positions count from 1, so 1 is the first "
                f"and {n} is the last." + self._example(example)
            )
        return int(position) - 1

    # ================================================================== looking
    @friendly
    def number_at(self, position=None):
        """The number at a position, counting from 1: number_at(1) is the first number.

        nums.number_at(1)
        Positions count from 1, the way people count and the way charts number the x axis.
        """
        return plain(self._values[self._check_position("number_at", position)])

    @friendly
    def show_first_numbers(self, how_many=5):
        """Show the first numbers. nums.show_first_numbers(5)"""
        n = check_whole("show_first_numbers", "show_first_numbers()", how_many, "nums.show_first_numbers(5)")
        print(np.array2string(self._values[:n], threshold=sys.maxsize, separator=", "))
        return None

    @friendly
    def show_last_numbers(self, how_many=5):
        """Show the last numbers. nums.show_last_numbers(5)"""
        n = check_whole("show_last_numbers", "show_last_numbers()", how_many, "nums.show_last_numbers(5)")
        print(np.array2string(self._values[self._values.size - min(n, self._values.size):],
                              threshold=sys.maxsize, separator=", "))
        return None

    # ================================================================== keeping and removing
    @friendly
    def keep_numbers(self, *extra, above=NOT_GIVEN, below=NOT_GIVEN, at_least=NOT_GIVEN, at_most=NOT_GIVEN,
                     equal_to=NOT_GIVEN, not_equal_to=NOT_GIVEN, between=NOT_GIVEN, one_of=NOT_GIVEN,
                     is_empty=NOT_GIVEN):
        """Keep only the numbers that match one condition. nums.keep_numbers(above=50)

        Conditions (the same as table.keep_rows): above, below, at_least, at_most, equal_to,
        not_equal_to, between=[low, high] (both ends included), one_of=[...], is_empty=True.
        Empty values only match is_empty.
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, is_empty=is_empty)
        cond, words = self._condition("keep_numbers", options, extra)
        return self._keep(f"keep numbers {words}", cond)

    @friendly
    def remove_numbers(self, *extra, above=NOT_GIVEN, below=NOT_GIVEN, at_least=NOT_GIVEN, at_most=NOT_GIVEN,
                       equal_to=NOT_GIVEN, not_equal_to=NOT_GIVEN, between=NOT_GIVEN, one_of=NOT_GIVEN,
                       is_empty=NOT_GIVEN):
        """Remove the numbers that match one condition (the opposite of keep_numbers).

        nums.remove_numbers(below=0)
        Same conditions as keep_numbers; between=[low, high] includes both ends.
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, is_empty=is_empty)
        cond, words = self._condition("remove_numbers", options, extra)
        return self._keep(f"remove numbers {words}", ~cond, kept_word=False)

    @friendly
    def keep_first_numbers(self, how_many=5):
        """Keep only the first numbers. nums.keep_first_numbers(10)"""
        n = check_whole("keep_first_numbers", "keep_first_numbers()", how_many, "nums.keep_first_numbers(10)")
        mask = np.arange(self._values.size) < n
        return self._keep(f"keep the first {amount(n)}", mask)

    @friendly
    def keep_last_numbers(self, how_many=5):
        """Keep only the last numbers. nums.keep_last_numbers(10)"""
        n = check_whole("keep_last_numbers", "keep_last_numbers()", how_many, "nums.keep_last_numbers(10)")
        mask = np.arange(self._values.size) >= self._values.size - n
        return self._keep(f"keep the last {amount(n)}", mask)

    @friendly
    def keep_random_numbers(self, how_many=5, seed=None):
        """Keep some numbers picked at random (they keep their order); a seed gives the same pick every time.

        nums.keep_random_numbers(10, seed=1)
        """
        example = "nums.keep_random_numbers(10, seed=1)"
        n = check_whole("keep_random_numbers", "keep_random_numbers()", how_many, example)
        seed = check_seed(seed, example)
        size = self._values.size
        if n > size:
            print(f"There are only {amount(size)}, so all of them are kept.")
            n = size
        picked = np.random.default_rng(seed).choice(size, size=n, replace=False)
        mask = np.zeros(size, dtype=bool)
        mask[picked] = True
        what = f"keep {n} random numbers" + (f" (seed {seed})" if seed is not None else "")
        return self._keep(what, mask)

    @friendly
    def remove_empty_numbers(self):
        """Remove the empty (missing) values. nums.remove_empty_numbers()"""
        return self._keep("remove empty values", ~self._empty_mask(), kept_word=False)

    @friendly
    def remove_duplicate_numbers(self):
        """Remove numbers that repeat an earlier number, so each number is left once, in its first place.

        nums.remove_duplicate_numbers()
        """
        import pandas as pd

        mask = ~pd.Series(self._values).duplicated().to_numpy()
        return self._keep("remove duplicate numbers", mask, kept_word=False)

    @friendly
    def fill_empty_numbers(self, with_value=None, with_the=None):
        """Fill the empty values with a number, or with the average, median, value above or value below.

        nums.fill_empty_numbers(with_value=0)   or   nums.fill_empty_numbers(with_the="average")
        with_the="line between neighbours" fills each gap on the straight line between the
        values just before and just after it; empty values at the start or end stay empty.
        """
        if (with_value is None) == (with_the is None):
            raise SimplydataError(
                ".fill_empty_numbers() needs exactly one of with_value= or with_the=."
                + self._example("fill_empty_numbers(with_value=0)")
                + self._example('fill_empty_numbers(with_the="average")')
            )
        if with_value is not None and not is_number(with_value):
            raise SimplydataError("with_value= needs a number, like with_value=0." + self._example("fill_empty_numbers(with_value=0)"))
        choice = None if with_the is None else str(with_the).strip().lower().replace("_", " ")
        if choice is not None:
            choice = choice.replace("neighbors", "neighbours")
        if choice is not None and choice not in FILL_CHOICES:
            raise choice_error("fill_empty_numbers", "with_the", with_the, FILL_CHOICES)
        empty = self._empty_mask()
        if not empty.any():
            print("There are no empty values to fill.")
            return self
        v = self._values.astype(float)
        if choice is None:
            new, how = np.where(empty, float(with_value), v), say(with_value)
        else:
            if choice in ("average", "median"):
                if empty.all():
                    raise SimplydataError(f"There are no numbers at all, so there is no {choice} to fill with.")
                value = np.nanmean(v) if choice == "average" else np.nanmedian(v)
                new = np.where(empty, value, v)
            elif choice == "line between neighbours":
                from .._stats import line_between_neighbours

                new = line_between_neighbours(np.where(empty, np.nan, v))
            else:
                import pandas as pd

                series = pd.Series(v)
                new = (series.ffill() if choice == "value above" else series.bfill()).to_numpy()
            how = f"the {choice}"
        left = int(np.isnan(new).sum())
        filled = int(empty.sum()) - left
        why = ("at the start or end, with a neighbour on one side only" if choice == "line between neighbours"
               else "nothing to copy from")
        print(f"Filled {amount(filled, 'empty value')}."
              + (f" {amount(left, 'value')} stayed empty ({why})." if left else ""))
        return self._change(f"fill empty values with {how}", new)

    # ================================================================== order
    @friendly
    def sort(self, biggest_first=False):
        """Put the numbers in order, smallest first; empty values go last. nums.sort(biggest_first=True)"""
        if not isinstance(biggest_first, bool):
            raise SimplydataError("biggest_first= needs True or False (no quotes)." + self._example("sort(biggest_first=True)"))
        empty = self._empty_mask()
        filled = np.sort(self._values[~empty], kind="stable")
        if biggest_first:
            filled = filled[::-1]
        new = np.concatenate([filled, self._values[empty]])
        return self._change("sort, biggest first" if biggest_first else "sort, smallest first", new)

    @friendly
    def reverse(self):
        """Turn the order round: the last number comes first. nums.reverse()"""
        return self._change("reverse", self._values[::-1].copy())

    @friendly
    def running_total(self):
        """Each number becomes the total so far (the cumulative sum): 1, 2, 3 becomes 1, 3, 6.

        nums.running_total()
        Empty values stay empty and are skipped in the total, like pandas cumsum().
        """
        empty = self._empty_mask()
        totals = np.cumsum(np.where(empty, 0, self._values))
        if empty.any():
            totals = np.where(empty, np.nan, totals)
        return self._change("running total", totals)

    # ================================================================== into a grid
    @friendly
    def number_grid(self, rows=None, columns=None):
        """Make a NEW grid from these numbers, filled row by row. grid = nums.number_grid(columns=3)

        Give rows= or columns= (or both); the numbers must fill the grid exactly.
        """
        from .numbers import number_grid

        return number_grid(self, rows=rows, columns=columns)
