"""The heart of simplydata numbers: a list of numbers, or a grid of them.

Numbers live in a NumPy array. Every step that changes them goes through self._change(...),
which remembers the numbers as they were before. That gives undo(), steps() and
start_again() for free, exactly like tables.

Two ways to do maths, on purpose:

* symbols make NEW numbers, like maths on paper:   bigger = nums + 5
* words CHANGE the numbers you have (with undo):   nums.add(5)

Both use the same calculation underneath, so they always give the same answer.

Rules shared by every word (the same as tables):

* an "empty" value is NaN (missing). Summaries skip empty values and say how many.
* a calculation with no answer (dividing by 0, the square root of a negative number,
  the log of 0) gives an empty value and a short note, never a crash or "inf".
* positions count from 1, like people do and like charts do (chart.lines([3, 5, 2])
  puts the first number at x = 1). Python's [ ] (which counts from 0) is not used.
"""

from __future__ import annotations

import numbers as _nb
import sys

import numpy as np

from .._errors import MissingWordError, SimplydataError, friendly, old_word, other_name_error, suggest

HISTORY_LIMIT = 30  # how many steps undo() can take back

# Words people may know from NumPy, pandas or tables -> the simplydata word.
HINTS = {
    "mean": "average", "avg": "average", "sum": "total", "min": "smallest", "max": "biggest",
    "minimum": "smallest", "maximum": "biggest", "std": "standard_deviation", "sd": "standard_deviation",
    "sqrt": "square_root", "abs": "absolute_value", "round": "round_to", "cumsum": "running_total",
    "multiply": "times", "divide": "divide_by", "plus": "add", "minus": "subtract", "pow": "power",
    "quantile": "percentile", "size": "count", "len": "count", "count_rows": "count",
    "keep_rows": "keep_numbers", "remove_rows": "remove_numbers", "filter": "keep_numbers",
    "keep_first_rows": "keep_first_numbers", "keep_last_rows": "keep_last_numbers",
    "keep_random_rows": "keep_random_numbers", "remove_empty_rows": "remove_empty_numbers",
    "remove_rows_with_empty_cells": "remove_empty_numbers", "count_empty_cells": "count_empty_numbers",
    "fill_empty_cells": "fill_empty_numbers", "show_first_rows": "show_first_numbers",
    "show_last_rows": "show_last_numbers", "as_pandas": "as_numpy", "as_polars": "as_numpy",
    "remove_empty": "remove_empty_numbers", "dropna": "remove_empty_numbers",
    "remove_duplicate_rows": "remove_duplicate_numbers", "unique": "remove_duplicate_numbers",
    "sort_by": "sort", "sorted": "sort", "reversed": "reverse", "fillna": "fill_empty_numbers",
    "transpose": "swap_rows_and_columns", "T": "swap_rows_and_columns", "flatten": "numbers",
    "ravel": "numbers", "reshape": "number_grid", "grid": "number_grid", "to_numpy": "as_numpy",
    "numpy": "as_numpy", "array": "as_numpy", "get": "number_at", "at": "number_at", "item": "number_at",
    "column_names": "count_columns", "count_numbers": "count", "head": "show_first_numbers",
    "tail": "show_last_numbers",
    "print": "show", "sqrt_": "square_root", "log10": "log", "log2": "log", "ln": "log",
}


class _NotGiven:
    def __repr__(self):
        return "not given"


NOT_GIVEN = _NotGiven()


def is_number(value) -> bool:
    """A real number (int, float, NumPy number), but not True/False."""
    return isinstance(value, _nb.Real) and not isinstance(value, (bool, np.bool_))


def plain(value):
    """A plain Python number (not a NumPy one)."""
    if hasattr(value, "item"):
        return value.item()
    return value


def say(value) -> str:
    """A number written for people: 5, 2.5, 0.001."""
    value = plain(value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def amount(n: int, word: str = "number") -> str:
    """1 number, 2 numbers."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def note_empty(n: int, why: str):
    """Print the note for values that became empty because a sum had no answer."""
    if n:
        print(f"Note: {why}, so {amount(n)} {'is' if n == 1 else 'are'} now empty.")


def _is_polars(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "polars"


def _is_pandas(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "pandas"


def _bad_item(item, step: str, example: str) -> SimplydataError:
    if isinstance(item, (str, bytes, np.str_)):
        text = item.decode() if isinstance(item, bytes) else str(item)
        hint = ""
        try:
            float(text)
            hint = f" Write numbers without quotes: {text.strip()} not \"{text}\"."
        except ValueError:
            pass
        return SimplydataError(f'{step} needs numbers, but "{text}" is text.{hint}\nFor example:  {example}')
    if isinstance(item, (bool, np.bool_)):
        return SimplydataError(
            f"{step} needs numbers, not True/False. Write 1 for True and 0 for False.\nFor example:  {example}"
        )
    return SimplydataError(f"{step} needs numbers, but it was given a {type(item).__name__}.\nFor example:  {example}")


def to_array(values, step: str, example: str) -> np.ndarray:
    """Turn what a user gives us into a NumPy array of numbers (whole numbers or decimals).

    None and NaN become empty values (NaN). Text, True/False and dates give a friendly error.
    """
    if isinstance(values, BaseNumbers):
        return values._values.copy()
    if isinstance(values, range):
        values = list(values)
    if _is_polars(values) or _is_pandas(values):
        if hasattr(values, "collect"):  # Polars LazyFrame
            values = values.collect()
        values = values.to_numpy()
    if isinstance(values, (str, bytes)) or is_number(values) or isinstance(values, (bool, np.bool_, dict, set)):
        raise SimplydataError(f"{step} needs a list of numbers in square brackets.\nFor example:  {example}")
    try:
        arr = np.asarray(values)
    except (ValueError, TypeError):
        raise SimplydataError(
            f"{step} needs rows of the same length (every row needs the same number of numbers).\n"
            f"For example:  {example}"
        ) from None
    kind = arr.dtype.kind
    if kind in "iu":
        return arr.astype(np.int64)
    if kind == "f":
        return arr.astype(np.float64)
    if kind == "b":
        raise _bad_item(True, step, example)
    if kind in "US":
        raise _bad_item(arr.ravel()[0], step, example)
    if kind in "mM":
        raise SimplydataError(f"{step} needs numbers, not dates or times.\nFor example:  {example}")
    if kind == "c":
        raise SimplydataError(f"{step} works with ordinary numbers, not complex numbers.\nFor example:  {example}")
    if kind == "O":
        flat = []
        whole = True
        for item in arr.ravel():
            if isinstance(item, (str, bytes, bool, np.bool_, np.str_)):
                raise _bad_item(item, step, example)
            if is_number(item):
                whole = whole and isinstance(item, _nb.Integral)
                flat.append(item)
                continue
            if isinstance(item, (list, tuple, np.ndarray)):
                raise SimplydataError(
                    f"{step} needs rows of the same length (every row needs the same number of numbers).\n"
                    f"For example:  {example}"
                )
            try:
                import pandas as pd

                missing = item is None or bool(pd.isna(item))
            except (TypeError, ValueError):
                missing = False
            if not missing:
                raise _bad_item(item, step, example)
            whole = False
            flat.append(np.nan)
        dtype = np.int64 if whole and flat else np.float64
        return np.array(flat, dtype=dtype).reshape(arr.shape)
    raise SimplydataError(f"{step} needs numbers.\nFor example:  {example}")


# ---------------------------------------------------------------------- the calculations
# Each gives back (result, number of values that became empty, why). Words and symbols both
# use these, so nums.add(5) and nums + 5 always agree.


def _filled(a, b, shape):
    return ~np.isnan(np.broadcast_to(np.asarray(a, dtype=float), shape)) & ~np.isnan(
        np.broadcast_to(np.asarray(b, dtype=float), shape)
    )


def calculate(op: str, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    with np.errstate(all="ignore"):
        if op == "add":
            return a + b, 0, ""
        if op == "subtract":
            return a - b, 0, ""
        if op == "times":
            return a * b, 0, ""
        if op == "divide_by":
            result = np.true_divide(a, b)
            zero = np.broadcast_to(b == 0, result.shape)
            n = int((zero & _filled(a, b, result.shape)).sum())
            result = np.where(zero, np.nan, result)
            return result, n, "dividing by 0 has no answer"
        if op == "power":
            return _power(a, b)
    raise ValueError(op)  # pragma: no cover


def _power(a, b):
    whole = a.dtype.kind in "iu" and b.dtype.kind in "iu"
    as_float = np.power(a.astype(float), b.astype(float))
    if whole and not (b < 0).any() and not (np.abs(as_float[np.isfinite(as_float)]) >= 2.0 ** 62).any():
        return np.power(a, b), 0, ""  # stays whole numbers when it safely can
    shape = as_float.shape
    a_b, b_b = np.broadcast_to(a, shape), np.broadcast_to(b, shape)
    zero_neg = (a_b == 0) & (b_b < 0)
    neg_frac = (a_b < 0) & (np.mod(b_b, 1) != 0)
    filled = _filled(a, b, shape)
    result = np.where(zero_neg | neg_frac, np.nan, as_float)
    n = int(((zero_neg | neg_frac) & filled).sum())
    whys = []
    if (zero_neg & filled).any():
        whys.append("0 to a minus power means dividing by 0, which has no answer")
    if (neg_frac & filled).any():
        whys.append("a negative number to a power with decimals (such as 0.5) has no answer")
    return result, n, " and ".join(whys)


SYMBOLS = {"add": "+", "subtract": "-", "times": "*", "divide_by": "/", "power": "**"}
UFUNCS = {np.add: "add", np.subtract: "subtract", np.multiply: "times", np.true_divide: "divide_by",
          np.power: "power"}


class BaseNumbers:
    """Shared by Numbers and NumberGrid."""

    _name = "nums"  # the name used in examples
    _noun = "numbers"
    # Old names that still work for one release (with a note the first time): old -> new.
    _OLD_WORDS = {"count_empty": "count_empty_numbers", "done": "as_numpy"}

    def __init__(self, values):
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_original", values)
        object.__setattr__(self, "_history", [])  # list of (what was done, values before)

    # ------------------------------------------------------------------ helpers for words
    def _change(self, what: str, values):
        """Replace the numbers with changed ones and remember the old ones for undo()."""
        self._history.append((what, self._values))
        if len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)
        object.__setattr__(self, "_values", np.asarray(values))
        return self

    @staticmethod
    def _wrap(values):
        """New Numbers or NumberGrid, whichever fits the shape."""
        from .numbers import NumberGrid, Numbers

        values = np.asarray(values)
        return NumberGrid(values) if values.ndim == 2 else Numbers(values)

    def _empty_mask(self):
        if self._values.dtype.kind == "f":
            return np.isnan(self._values)
        return np.zeros(self._values.shape, dtype=bool)

    def _example(self, code: str) -> str:
        return f"\nFor example:  {self._name}.{code}"

    def _operand(self, other, step: str):
        """The other side of a sum, checked: one number, or numbers that line up with these."""
        example = f"{self._name}.{step}(5)"
        if is_number(other):
            return np.asarray(other)
        if other is None or isinstance(other, (str, bytes, bool, np.bool_)):
            if isinstance(other, str):
                try:
                    float(other)
                    raise SimplydataError(
                        f"Write numbers without quotes: {self._name}.{step}({other.strip()}), "
                        f'not {self._name}.{step}("{other}").'
                    )
                except ValueError:
                    pass
            raise SimplydataError(f".{step}() needs a number, or a list of numbers." + self._example(f"{step}(5)"))
        values = to_array(other, f".{step}()", example)
        mine = self._values.shape
        if values.ndim == 0:
            return values
        if len(mine) == 1:
            if values.ndim == 1 and values.shape == mine:
                return values
            if values.ndim == 2 and values.shape[1] == mine[0]:
                return values  # numbers + grid: added to each row of the grid
            raise SimplydataError(
                f".{step}() needs one number, or {amount(mine[0])} (one for each of yours), "
                f"but it was given {_describe(values)}." + self._example(f"{step}(5)")
            )
        if values.shape == mine or (values.ndim == 1 and values.shape[0] == mine[1]):
            return values
        raise SimplydataError(
            f".{step}() needs one number, a grid of the same size ({mine[0]} rows, {mine[1]} columns), "
            f"or {amount(mine[1])} (one for each column), but it was given {_describe(values)}."
            + self._example(f"{step}(5)")
        )

    def _sum_word(self, op: str, other):
        """Do a sum with a word: changes these numbers, with undo."""
        b = self._operand(other, op)
        result, n, why = calculate(op, self._values, b)
        if result.shape != self._values.shape:
            raise SimplydataError(
                f"{self._name}.{op}() would change the shape of your {self._noun}. "
                f"Use the symbol instead to make new ones, for example:  new = {self._name} {SYMBOLS[op]} other"
            )
        note_empty(n, why)
        word = op.replace("_", " ")
        return self._change(f"{word} {_say_operand(other)}", result)

    def _sum_symbol(self, op: str, other, reverse: bool = False):
        """Do a sum with a symbol: makes new numbers, these stay as they are."""
        if not (is_number(other) or isinstance(other, (BaseNumbers, list, tuple, np.ndarray, range))
                or _is_pandas(other) or _is_polars(other)):
            return NotImplemented
        b = self._operand(other, op)
        a = self._values
        result, n, why = calculate(op, b, a) if reverse else calculate(op, a, b)
        note_empty(n, why)
        return self._wrap(result)

    # ------------------------------------------------------------------ history words
    @friendly
    def undo(self):
        """Take the last step back. nums.undo()"""
        if not self._history:
            print(f"Nothing to undo: these {self._noun} have no steps yet.")
            return self
        what, before = self._history.pop()
        object.__setattr__(self, "_values", before)
        print(f"Undone: {what}")
        return self

    @friendly
    def steps(self):
        """List every step you did, in order. nums.steps()"""
        if not self._history:
            print("No steps yet.")
        else:
            for i, (what, _) in enumerate(self._history, 1):
                print(f"{i}. {what}")
        return self

    @friendly
    def start_again(self):
        """Go back to the numbers as they were at the start. nums.start_again()"""
        return self._change("start again", self._original)

    # ------------------------------------------------------------------ looking words
    @friendly
    def show(self):
        """Show every number. nums.show()"""
        print(np.array2string(self._values, threshold=sys.maxsize, separator=", "))
        return None

    @friendly
    def count(self):
        """How many numbers there are (empty ones included). nums.count()"""
        return int(self._values.size)

    @friendly
    def count_empty_numbers(self):
        """How many values are empty (missing). nums.count_empty_numbers()"""
        return int(self._empty_mask().sum())

    # ------------------------------------------------------------------ finishing words
    @friendly
    def as_numpy(self):
        """Give back the numbers as a plain NumPy array, for other Python tools. array = nums.as_numpy()"""
        return self._values.copy()

    @friendly
    def table(self, name="value"):
        """Turn the numbers into a simplydata table. nums.table(name="marks")"""
        from ..table import Table

        return Table(self._as_pandas(name=name))

    @friendly
    def chart(self):
        """Start a chart from these numbers. chart = nums.chart()"""
        from ..chart import chart

        return chart(self)

    @friendly
    def save(self, file_name):
        """Save the numbers; the end of the file name decides the type. nums.save("nums.csv")

        File types: .csv, .tsv, .xlsx (Excel), .parquet and .json, the same as tables.
        """
        self.table().save(file_name)
        return self

    # ------------------------------------------------------------------ symbols (new numbers)
    def __add__(self, other):
        return self._sum_symbol("add", other)

    def __radd__(self, other):
        return self._sum_symbol("add", other, reverse=True)

    def __sub__(self, other):
        return self._sum_symbol("subtract", other)

    def __rsub__(self, other):
        return self._sum_symbol("subtract", other, reverse=True)

    def __mul__(self, other):
        return self._sum_symbol("times", other)

    def __rmul__(self, other):
        return self._sum_symbol("times", other, reverse=True)

    def __truediv__(self, other):
        return self._sum_symbol("divide_by", other)

    def __rtruediv__(self, other):
        return self._sum_symbol("divide_by", other, reverse=True)

    def __pow__(self, other):
        return self._sum_symbol("power", other)

    def __rpow__(self, other):
        return self._sum_symbol("power", other, reverse=True)

    def __neg__(self):
        return self._wrap(-self._values)

    def __pos__(self):
        return self._wrap(self._values.copy())

    def __abs__(self):
        return self._wrap(np.abs(self._values))

    def __matmul__(self, other):
        raise SimplydataError(
            "Matrix multiplication (@) is planned for a later version of simplydata (v0.7, with "
            "matrices and linear algebra).\nFor now use done() to get a NumPy array: "
            f"{self._name}.as_numpy() @ other.as_numpy()"
        )

    __rmatmul__ = __matmul__

    # ------------------------------------------------------------------ Python and NumPy details
    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values.tolist())

    def __array__(self, dtype=None, copy=None):
        return np.array(self._values, dtype=dtype, copy=True)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy functions work too: np.sin(nums) gives new numbers."""
        if method == "__call__" and not kwargs and len(inputs) == 2 and ufunc in UFUNCS:
            if inputs[0] is self:
                return self._sum_symbol(UFUNCS[ufunc], inputs[1])
            return self._sum_symbol(UFUNCS[ufunc], inputs[0], reverse=True)
        if "out" in kwargs:
            return NotImplemented
        args = [x._values if isinstance(x, BaseNumbers) else x for x in inputs]
        result = getattr(ufunc, method)(*args, **kwargs)
        if (method == "__call__" and isinstance(result, np.ndarray) and result.ndim in (1, 2)
                and result.dtype.kind in "iuf"):
            return self._wrap(result)
        return result

    def __getitem__(self, position):
        raise SimplydataError(self._picking_help())

    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._OLD_WORDS:
            return old_word(self, name, self._OLD_WORDS)
        known = self._word_names()
        pointer = other_name_error("numbers", name, known, prefix=self._name)
        if pointer is not None:
            raise pointer
        target = HINTS.get(name)
        if target is None and name.endswith("_of") and name[:-3] in known:
            target = name[:-3]
            return_msg = (
                f"simplydata has no step called .{name}() for {self._noun}.\n"
                f"Tables write {name}(\"column\"), but {self._noun} have no columns to name, so write "
                f"{self._name}.{target}()."
            )
            raise MissingWordError(return_msg)
        if target in known:
            raise MissingWordError(
                f"simplydata has no step called .{name}() for {self._noun}.\n"
                f"Did you mean {self._name}.{target}()?\nEvery step for {self._noun}: " + ", ".join(known)
            )
        lines = [f"simplydata has no step called .{name}() for {self._noun}."]
        guesses = suggest(name, known)
        if guesses:
            lines.append("Did you mean " + " or ".join(f".{g}()" for g in guesses) + "?")
        lines.append(f"Every step for {self._noun}: " + ", ".join(known))
        raise MissingWordError("\n".join(lines))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise SimplydataError(
            f"Steps are written with brackets, not with =.\n"
            f"Instead of {self._name}.{name} = ..., write a step such as {self._name}.add(5)"
        )


def _describe(values: np.ndarray) -> str:
    if values.ndim == 1:
        return amount(values.shape[0])
    return f"a grid of {values.shape[0]} rows and {values.shape[1]} columns"


def _say_operand(other) -> str:
    if is_number(other):
        return say(other)
    if isinstance(other, BaseNumbers):
        return f"other {other._noun}"
    try:
        values = list(other)
    except TypeError:
        return "other numbers"
    if len(values) <= 6 and all(is_number(v) for v in values):
        return "[" + ", ".join(say(v) for v in values) + "]"
    return "other numbers"


def check_whole(step: str, option: str, value, example: str, smallest: int = 0) -> int:
    if not isinstance(value, _nb.Integral) or isinstance(value, (bool, np.bool_)) or value < smallest:
        raise SimplydataError(f"{option} needs a whole number{' from 1 up' if smallest == 1 else ''}, like 5."
                              f"\nFor example:  {example}")
    return int(value)


def check_seed(seed, example: str):
    if seed is not None and (not isinstance(seed, _nb.Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0):
        raise SimplydataError(
            "seed= needs a whole number, like seed=1. The same seed gives the same random numbers again."
            f"\nFor example:  {example}"
        )
    return None if seed is None else int(seed)
