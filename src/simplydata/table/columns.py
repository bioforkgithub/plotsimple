"""Sorting, new columns, combining tables, reshaping.

Every word works on pandas and Polars tables alike, because the work is done with
narwhals. Only add_rows_from() uses each engine's own code (so that whole numbers and
decimal numbers can be stacked without fuss).

Calculator-style formulas, add_column("total", "maths + english"), are never run as
Python code. The text is read with Python's ast module and only these pieces are allowed:
column names, numbers, + - * / ** // %, brackets, a minus sign in front, the constant pi,
the functions round, abs, sqrt, log, exp, sin, cos and tan, the special functions erf,
erfc, gamma, lgamma, digamma, beta and besselJ (named and ordered as in R, worked out by
SciPy), and the summary words average, median, total, smallest, biggest and count.
Everything else is refused with a friendly message. Rules in
add_group_column(rules=) are read by the shared condition reader in conditions.py (the
same one keep_rows() uses), with comparisons, and / or / not, in [...] and is empty added.

A few words need each engine's own code, because narwhals has no common word for them:
the ISO week number, splitting text into columns, pulling out part of a text with a
pattern, and rolling medians / smallest / biggest values. Both versions are tested
against plain pandas.
"""

from __future__ import annotations

import ast
import datetime as _dt
import keyword
import math
import numbers
import re
import warnings

import narwhals as nw

from .._data import load_native
from .._errors import SimplydataError, choice_error, column_error, friendly, suggest

# ---------------------------------------------------------------------------- formulas

FORMULA_HELP = (
    "A formula can use:\n"
    "  column names               marks, hours  (names with spaces go in backticks: `max temp`)\n"
    "  numbers                    2, 100, 0.5\n"
    "  + - * /                    add, subtract, multiply, divide\n"
    "  **                         power: height ** 2\n"
    "  //  %                      whole-number division and remainder\n"
    "  ( )                        brackets\n"
    "  round(x)  round(x, 1)      round, or round to 1 decimal place\n"
    "  abs(x)  sqrt(x)            size without the minus sign, square root\n"
    "  log(x)  log(x, 10)         natural log, log to base 10\n"
    "  exp(x)                     e to the power x\n"
    "  sin(x)  cos(x)  tan(x)     sine, cosine, tangent (x in radians); pi is 3.14159...\n"
    "  erf(x)  erfc(x)            error function, and 1 - erf(x)\n"
    "  gamma(x)  lgamma(x)        gamma function, and the log of its size (as R)\n"
    "  digamma(x)  beta(a, b)     digamma function, beta function (as R)\n"
    "  besselJ(x, nu)             Bessel function of the first kind, order nu (as R)\n"
    "  average(x)  median(x)      summary words: one value for the whole column,\n"
    "  total(x)  count(x)         or for each group with for_each=\n"
    "  smallest(x)  biggest(x)    (count counts the cells that are not empty)\n"
    'Example: table.add_column("marks_per_hour", "marks / hours")   (with sdata.example("marks"))\n'
    'Example: table.add_column("share", "marks / total(marks) * 100", for_each="class")'
)

FUNCTIONS = {"round": (1, 2), "abs": (1, 1), "sqrt": (1, 1), "log": (1, 2), "exp": (1, 1),
             "sin": (1, 1), "cos": (1, 1), "tan": (1, 1), "erf": (1, 1), "erfc": (1, 1), "gamma": (1, 1),
             "lgamma": (1, 1), "digamma": (1, 1), "beta": (2, 2), "besselJ": (2, 2)}
# functions worked out with SciPy / NumPy on the numbers themselves (narwhals has no word for them)
SPECIAL_FUNCTIONS = ("tan", "erf", "erfc", "gamma", "lgamma", "digamma", "beta", "besselJ")
FUNCTION_SPELLINGS = {"besselj": "besselJ"}  # both spellings work
CONSTANTS = {"pi": math.pi}
# summary words inside formulas -> the narwhals method that works them out
SUMMARY_WORDS = {"average": "mean", "median": "median", "total": "sum",
                 "smallest": "min", "biggest": "max", "count": "count"}
# names people bring from other tools -> the simplydata word
OTHER_TOOL_NAMES = {"mean": "average", "avg": "average", "sum": "total", "min": "smallest",
                    "max": "biggest", "n": "count", "len": "count", "length": "count",
                    "gammaln": "lgamma", "loggamma": "lgamma", "psi": "digamma", "jv": "besselJ",
                    "power": "**", "pow": "**"}
MAX_FORMULA_LENGTH = 500

KEEP_CHOICES = {
    "matching rows": "inner",
    "all rows from this table": "left",
    "all rows from both": "full",
}
JOIN_SUFFIX = "_other"


def _formula_error(problem: str) -> SimplydataError:
    return SimplydataError(f"{problem}\n\n{FORMULA_HELP}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _divide(top, bottom, how="/"):
    """Divide, but give an empty cell when dividing by zero."""
    if how == "/":
        result = top / bottom
    elif how == "//":
        result = top // bottom
    else:
        result = top % bottom
    return nw.when(bottom != 0).then(result)


def special_function(name: str, *values):
    """erf(x), gamma(x), beta(a, b), besselJ(x, nu) ... on NumPy arrays, with SciPy.

    Names and the order of the inputs are those of R: lgamma is the log of the size of
    gamma (SciPy gammaln), besselJ(x, nu) is SciPy jv(nu, x).
    """
    import numpy as np
    from scipy import special

    values = [np.asarray(v, dtype=float) for v in values]
    with np.errstate(all="ignore"):
        if name == "tan":
            return np.tan(values[0])
        if name == "beta":
            return special.beta(values[0], values[1])
        if name == "besselJ":
            return special.jv(values[1], values[0])
        found = {"erf": special.erf, "erfc": special.erfc, "gamma": special.gamma,
                 "lgamma": special.gammaln, "digamma": special.digamma}[name]
        return found(values[0])


class _FormulaReader:
    """Turns formula text into a narwhals expression. The text is never run.

    After read():
      used         every column the formula names
      number_used  the columns that must hold numbers (count(name) may count text)
      summaries    (temporary column, expression inside, summary word) for each summary
                   word such as total(marks); _apply_summaries() works them out
    """

    def __init__(self, text, columns):
        self.text = text
        self.columns = list(columns)
        self.used: list = []
        self.number_used: list = []
        self.placeholders: dict = {}
        self.summaries: list = []
        self.specials: list = []  # (temporary column, expressions inside, function name)
        self.steps: list = []  # summaries and special functions, inner ones first
        self._in_summary = False

    def _error(self, problem):
        return _formula_error(problem)

    def _no_column_error(self):
        return self._error(
            "The formula has no column in it. For the same value in every row use value=,\n"
            'for example table.add_column("school", value="Hill School").'
        )

    def _what(self):
        return "formula"

    def read(self):
        text = self.text
        what = self._what()
        if not isinstance(text, str) or not text.strip():
            raise self._error(f'The {what} should be some text, for example "marks / hours".')
        if len(text) > MAX_FORMULA_LENGTH:
            raise self._error(f"That {what} is very long. Keep it under {MAX_FORMULA_LENGTH} letters.")

        def backtick(match):
            name = match.group(1)
            if name not in self.columns:
                raise column_error(name, self.columns)
            return self._placeholder(name)

        text = re.sub(r"`([^`]*)`", backtick, text)
        if "`" in text:
            raise self._error(
                "A backtick ` is missing. Put a column name with spaces between two backticks: `max temp`."
            )
        text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
        # columns named like Python words (for example "class") still work without backticks;
        # text in quotes (in rules: class == "A") is left alone
        for name in self.columns:
            if isinstance(name, str) and keyword.iskeyword(name):
                pattern = rf"""('[^']*'|"[^"]*")|(?<![\w.]){re.escape(name)}(?!\w)"""
                text = re.sub(pattern, lambda m, n=name: m.group(1) or self._placeholder(n), text)
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            raise self._error(
                f'simplydata cannot read the {what} "{self.text}". Check the brackets and signs.'
            ) from None
        except (ValueError, RecursionError, MemoryError):
            raise self._error(f'simplydata cannot read the {what} "{self.text}".') from None
        expr = self._top(tree.body)
        if not self.used:
            raise self._no_column_error()
        return expr

    def _top(self, node):
        return self._node(node)

    def _placeholder(self, name):
        key = f"_simplydata_column_{len(self.placeholders)}_"
        self.placeholders[key] = name
        return key

    def _snippet(self, node):
        text = ast.unparse(node)
        for key, name in self.placeholders.items():
            text = text.replace(key, f"`{name}`")
        return text

    def _column_name(self, node):
        """The column a bare name stands for, or None."""
        if isinstance(node, ast.Name):
            if node.id in self.placeholders:
                return self.placeholders[node.id]
            if node.id in self.columns:
                return node.id
        return None

    def _column(self, name, needs_number=True):
        if name not in self.used:
            self.used.append(name)
        if needs_number and name not in self.number_used:
            self.number_used.append(name)
        return nw.col(name)

    def _node(self, node):
        if isinstance(node, ast.Constant):
            if not _is_number(node.value):
                raise self._error(
                    f"{node.value!r} is not allowed in a {self._what()}: only numbers and column names are."
                )
            return nw.lit(node.value)
        if isinstance(node, ast.Name):
            name = self._column_name(node)
            if name is not None:
                return self._column(name)
            if node.id in CONSTANTS:
                return nw.lit(CONSTANTS[node.id])
            if node.id in FUNCTIONS or node.id in SUMMARY_WORDS:
                raise self._error(f"{node.id} needs brackets after it, for example {node.id}(marks).")
            raise column_error(node.id, self.columns)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self._node(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.BitXor):
                raise self._error("For a power use ** instead of ^, for example height ** 2.")
            kinds = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
            if not isinstance(op, kinds):
                raise self._error(f'The sign in "{self._snippet(node)}" is not allowed in a {self._what()}.')
            left, right = self._node(node.left), self._node(node.right)
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return _divide(left, right, "/")
            if isinstance(op, ast.FloorDiv):
                return _divide(left, right, "//")
            if isinstance(op, ast.Mod):
                return _divide(left, right, "%")
            return left.cast(nw.Float64) ** right.cast(nw.Float64)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Attribute):
            raise self._error(
                f'"{self._snippet(node)}" is not allowed: a {self._what()} cannot use a dot after a name.'
            )
        if isinstance(node, (ast.Compare, ast.BoolOp)) or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            raise self._error(
                f'"{self._snippet(node)}" is a condition (yes or no), which a formula cannot use.\n'
                'Conditions go in rules=, for example table.add_group_column("result", rules={"marks >= 60": "pass"}, '
                'otherwise="fail").'
            )
        raise self._error(f'"{self._snippet(node)}" is not allowed in a {self._what()}.')

    def _call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in SUMMARY_WORDS:
            return self._summary(node)
        if isinstance(node.func, ast.Name) and node.func.id in FUNCTION_SPELLINGS:
            node.func.id = FUNCTION_SPELLINGS[node.func.id]
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else self._snippet(node.func)
            if OTHER_TOOL_NAMES.get(name) == "**":
                hint = " For a power write x ** 2."
            elif name in OTHER_TOOL_NAMES:
                hint = f" In simplydata it is called {OTHER_TOOL_NAMES[name]}()."
            else:
                guess = suggest(name, [*FUNCTIONS, *SUMMARY_WORDS], n=1)
                hint = f" Did you mean {guess[0]}()?" if guess else ""
            raise self._error(f'"{name}()" is not allowed in a {self._what()}.{hint}')
        name = node.func.id
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise self._error(f"Write {name}() simply, like {name}(marks).")
        low, high = FUNCTIONS[name]
        if not low <= len(node.args) <= high:
            needs = "one thing" if high == 1 else "two things" if low == 2 else "one or two things"
            given = f"{len(node.args)} thing" + ("" if len(node.args) == 1 else "s")
            raise self._error(f"{name}() was given {given} inside the brackets; it takes {needs}.")
        if name in SPECIAL_FUNCTIONS:
            inside = [self._node(a) for a in node.args]
            key = f"_simplydata_special_{len(self.steps)}_"
            self.specials.append((key, inside, name))
            self.steps.append(("special", key, inside, name))
            return nw.col(key)
        inner = self._node(node.args[0])
        if name == "abs":
            return inner.abs()
        if name == "sqrt":
            return inner.cast(nw.Float64).sqrt()
        if name == "exp":
            return inner.cast(nw.Float64).exp()
        if name == "sin":
            return inner.cast(nw.Float64).sin()
        if name == "cos":
            return inner.cast(nw.Float64).cos()
        second = node.args[1] if len(node.args) == 2 else None
        if second is not None and not (isinstance(second, ast.Constant) and _is_number(second.value)):
            raise self._error(
                f"The second thing in {name}() must be a plain number, for example {name}(marks, 1)."
            )
        if name == "round":
            digits = 0 if second is None else second.value
            if not isinstance(digits, int) or digits < 0:
                raise self._error("round(x, 1) needs a whole number of decimal places, like 0, 1 or 2.")
            # Halves go to the even number (2.5 -> 2, 3.5 -> 4), like R, NumPy and the
            # IEEE standard. This avoids a small upward bias when many values are rounded.
            return inner.round(digits)
        if second is None:  # log
            return inner.cast(nw.Float64).log()
        if second.value <= 0 or second.value == 1:
            raise self._error("The base of log() must be a positive number other than 1, for example log(x, 10).")
        return inner.cast(nw.Float64).log(base=float(second.value))

    def _summary(self, node):
        """average(marks), total(marks) ...: one value for the whole column (or each group)."""
        name = node.func.id
        if self._in_summary:
            raise self._error(f"{name}() cannot be inside another summary word. Use one at a time, like total(marks).")
        if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
            raise self._error(f"{name}() takes one column, for example {name}(marks).")
        arg = node.args[0]
        self._in_summary = True
        try:
            column = self._column_name(arg)
            if name == "count" and column is not None:
                inner = self._column(column, needs_number=False)  # count may count text cells
            else:
                inner = self._node(arg)
        finally:
            self._in_summary = False
        key = f"_simplydata_summary_{len(self.steps)}_"
        self.summaries.append((key, inner, name))
        self.steps.append(("summary", key, inner, name))
        return nw.col(key)


def _apply_summaries(frame, reader, keys):
    """Add the summary and special-function columns a formula asked for (summaries per group
    when keys are given), inner ones first, so gamma(total(x)) and total(gamma(x)) both work.

    Gives (frame, temporary column names to drop afterwards). Empty cells are left out of
    every summary, as in pandas and Polars: count() counts only the cells that are filled.
    A special function gives an empty cell where its answer is not a number.
    """
    import numpy as np

    temporary = []
    for kind, key, inner, word in reader.steps:
        if kind == "summary":
            frame = frame.with_columns(inner.alias(key + "in_"))
            expr = getattr(nw.col(key + "in_"), SUMMARY_WORDS[word])()
            frame = frame.with_columns((expr.over(*keys) if keys else expr).alias(key))
            temporary += [key + "in_", key]
            continue
        frame = nw.maybe_reset_index(frame)
        names = [f"{key}in{i}_" for i in range(len(inner))]
        frame = frame.with_columns(e.cast(nw.Float64).alias(n) for e, n in zip(inner, names))
        arrays = [np.asarray(frame[n].to_numpy(), dtype=float) for n in names]
        values = special_function(word, *arrays)
        frame = frame.with_columns(nw.new_series(key, values, nw.Float64, backend=frame.implementation))
        frame = frame.with_columns(nw.col(key).fill_nan(None))
        temporary += names + [key]
    return frame, temporary


def _dtype_kind(dtype) -> str:
    """The kind of a column in plain words (as in rows.py)."""
    if dtype == nw.Boolean:
        return "true/false"
    if dtype.is_numeric():
        return "number"
    if dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum:
        return "text"
    if dtype == nw.Date or dtype == nw.Datetime:
        return "date"
    return "other"


# ---------------------------------------------------------------------------- helpers

def _as_list(columns):
    if isinstance(columns, (list, tuple)):
        out = []
        for c in columns:
            out.extend(_as_list(c))
        return out
    return [columns]


def _plain(value):
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _blank_to_empty(expr):
    """Text that is only spaces or nothing becomes an empty cell."""
    return nw.when(expr.str.strip_chars() != "").then(expr)


def _escape_mark(text):
    """Escape a plain piece of text for use inside a pattern (both pandas and Polars read it)."""
    return "".join("\\" + ch if ch in "\\.^$|?*+()[]{}" else ch for ch in text)


def _whole_number(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _rows_word(n):
    return f"{n} row" if n == 1 else f"{n} rows"


def _cells_word(n):
    return f"{n} cell" if n == 1 else f"{n} cells"


def _value_kind(value):
    """The kind of one value given back by a function in by_running=."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (_dt.date, _dt.datetime)):
        return "date"
    return "other"


def _plain_cell(value):
    """A cell as plain Python, the same on both engines: empty cells become None."""
    if value is None:
        return None
    try:
        if value != value:  # NaN, NaT, pandas NA
            return None
    except (TypeError, ValueError):
        if str(value) in ("<NA>", "NaT"):
            return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError, TypeError):
            return value
    return value


ADD_COLUMN_WAYS = (
    '  a formula          table.add_column("marks_per_hour", "marks / hours")\n'
    '  by_adding=         table.add_column("study_and_marks", by_adding=["hours", "marks"])\n'
    "  by_subtracting=, by_multiplying=, by_dividing=   the same way\n"
    '  by_joining_text=   table.add_column("name_and_class", by_joining_text=["name", "class"], separator=" ")\n'
    '  from_column=       table.add_column("half", from_column="marks", divide_by=2)\n'
    '  value=             table.add_column("school", value="Hill School")\n'
    '  by_running=        table.add_column("per_hour", by_running=per_hour)   (a function you wrote)\n'
    '(all with sdata.example("marks"))'
)
FROM_COLUMN_EXAMPLE = (
    'For example: table.add_column("half", from_column="marks", divide_by=2)   (with sdata.example("marks"))'
)
DATE_PARTS = ("year", "quarter", "month", "month name", "week", "day", "weekday", "weekday name",
              "day of year", "hour", "minute")
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December")
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ROLLING_SHOWS = ("average", "total", "median", "smallest", "biggest", "standard deviation")
TEXT_PART_WAYS = (
    '  first_characters=3      the first 3 characters     table.add_text_part_column("initial", from_column="name", first_characters=1)\n'
    "  last_characters=3       the last 3 characters\n"
    "  characters=[2, 4]       characters 2 to 4 (counting from 1)\n"
    '  before=" "              the text before the first " "\n'
    '  after=" "               the text after the first " "\n'
    r'  pattern=r"\d+"          the first piece that fits a regular expression' "\n"
    '(with sdata.example("marks"))'
)


class ColumnWords:
    # ------------------------------------------------------------------ small helpers
    def _need_numbers(self, *names, step="add_column"):
        for name in names:
            dtype = self._frame.schema[name]
            if not dtype.is_numeric():
                numbers = [c for c, d in self._frame.schema.items() if d.is_numeric()]
                kind = "text" if dtype == nw.String else "values that are not numbers"
                raise SimplydataError(
                    f'Column "{name}" holds {kind}, so .{step}() cannot do sums with it.\n'
                    "Columns with numbers: " + (", ".join(map(str, numbers)) if numbers else "none")
                )

    @staticmethod
    def _check_new_name(name, step):
        if not isinstance(name, str) or not name.strip():
            raise SimplydataError(
                f'.{step}() needs a name for the new column first, for example table.{step}("total", ...).'
            )

    def _put_column(self, name, expr, what):
        """Add (or replace) a column. Impossible answers (like 1/0 or sqrt(-1)) become empty cells."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame = self._frame.with_columns(expr.alias(name))
        return self._finish(name, frame, what)

    def _finish(self, name, frame, what):
        """Record a frame that has the new column. Impossible answers (1/0, sqrt(-1)) become empty cells."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if frame.schema[name].is_float():
                frame = frame.with_columns(nw.when(nw.col(name).is_finite()).then(nw.col(name)).alias(name))
        return self._change(what, frame)

    def _kind(self, column):
        return _dtype_kind(self._frame.schema[column])

    def _text_of(self, step, column, example):
        """The column as text, or a friendly error if it does not hold text."""
        kind = self._kind(column)
        if kind == "text":
            return nw.col(column).cast(nw.String)
        hint = f' To use it as text, first write table.make_text("{column}").' if kind in ("number", "date") else ""
        raise SimplydataError(f'.{step}() works on text, but "{column}" holds {kind} values.{hint}\n{example}')

    def _group_keys(self, for_each, example):
        """for_each= as a list of column names (empty when not given)."""
        if for_each is None:
            return []
        keys = _as_list(for_each)
        if not keys or not all(isinstance(k, str) for k in keys):
            raise SimplydataError(f"for_each= should be a column name or a list of column names.\n{example}")
        self._need_columns(*keys)
        return keys

    def _order_columns(self, sorted_by, example):
        if sorted_by is None:
            return []
        cols = _as_list(sorted_by)
        if not cols or not all(isinstance(c, str) for c in cols):
            raise SimplydataError(f"sorted_by= should be a column name, such as a date or a year.\n{example}")
        self._need_columns(*cols)
        return cols

    def _row_after_row(self, name, keys, order, make, what):
        """Work out a column row after row: inside each for_each group, in the order of sorted_by.

        The rows of the table keep their place; only the calculation follows sorted_by.
        make(frame, over) gets the frame in calculation order and adds the column `name`;
        over(expr) makes an expression work inside each group, in that order. (narwhals
        needs order_by= for this on pandas: without it, rolling values per group come out
        in the wrong rows.)
        """
        index, step = "_simplydata_row_", "_simplydata_step_"
        frame = self._frame.with_row_index(index)
        if order:
            frame = nw.maybe_reset_index(frame.sort([*order, index], nulls_last=True))
        frame = frame.with_row_index(step)

        def over(expr):
            return expr.over(*keys, order_by=step) if keys else expr

        frame = make(frame, over)
        if order:
            frame = frame.sort(index)
        return self._finish(name, frame.drop([index, step]), what)

    def _order_words(self, keys, order):
        group = f" (separately for each {', '.join(map(str, keys))})" if keys else ""
        how = f"in order of {', '.join(map(str, order))}" if order else "in the order the rows are in now"
        return group, how

    def _number_or_column(self, value, option):
        if isinstance(value, str):
            self._need_columns(value)
            self._need_numbers(value)
            return nw.col(value)
        if not _is_number(value):
            raise SimplydataError(f"{option}= should be a number or a column name, not {value!r}.\n{FROM_COLUMN_EXAMPLE}")
        return nw.lit(value)

    def _other_frame(self, other, step):
        """Another table (simplydata, pandas, Polars or a file name), in this table's engine."""
        if other is None:
            raise SimplydataError(f'.{step}() needs another table, for example table.{step}("other_file.xlsx").')
        if getattr(other, "_is_simplydata_table", False):
            frame = other._frame
        else:
            frame = nw.from_native(load_native(other), eager_only=True)
        if frame.implementation == self._frame.implementation:
            return frame
        if self._frame.implementation.is_polars():
            import polars as pl

            return nw.from_native(pl.from_pandas(frame.to_pandas()), eager_only=True)
        return nw.from_native(frame.to_pandas(), eager_only=True)

    # ------------------------------------------------------------------ sorting
    @friendly
    def sort_by(self, *columns, biggest_first=False):
        """Put the rows in order of one or more columns; empty cells go last.

        Numbers go smallest first and text goes A to Z. biggest_first=True turns this round:
        biggest number first, and text Z to A. Rows that tie keep the order they had.
        table.sort_by("marks", biggest_first=True)   (with sdata.example("marks"))
        """
        names = _as_list(columns)
        if not names:
            raise SimplydataError(
                'sort_by() needs a column name, for example table.sort_by("marks")   (with sdata.example("marks"))'
            )
        if not isinstance(biggest_first, bool):
            raise SimplydataError("biggest_first= should be True or False.")
        self._need_columns(*names)
        order_column = "_simplydata_order_"
        frame = (
            self._frame.with_row_index(order_column)
            .sort([*names, order_column], descending=[biggest_first] * len(names) + [False], nulls_last=True)
            .drop(order_column)
        )
        order = "biggest first" if biggest_first else "smallest first"
        return self._change(f"sort by {', '.join(map(str, names))} ({order})", frame)

    # ------------------------------------------------------------------ new columns
    def _work_out(self, step, name, formula=None, by_adding=None, by_subtracting=None, by_multiplying=None,
                  by_dividing=None, by_joining_text=None, separator=None, by_running=None, for_each=None,
                  from_column=None, plus=None, minus=None, times=None, divide_by=None, value=None):
        """The new values for add_column() and change_column(): (frame with the column, words for steps())."""
        lists = {"by_adding": by_adding, "by_subtracting": by_subtracting,
                 "by_multiplying": by_multiplying, "by_dividing": by_dividing}
        numbers_given = {"plus": plus, "minus": minus, "times": times, "divide_by": divide_by}
        given_numbers = [k for k, v in numbers_given.items() if v is not None]
        ways = (["a formula"] if formula is not None else []) + [f"{k}=" for k, v in lists.items() if v is not None]
        ways += (["from_column="] if from_column is not None else []) + (["value="] if value is not None else [])
        ways += (["by_joining_text="] if by_joining_text is not None else [])
        ways += (["by_running="] if by_running is not None else [])
        if not ways and given_numbers:
            raise SimplydataError(
                f"{given_numbers[0]}= needs from_column= to say which column to start from.\n{FROM_COLUMN_EXAMPLE}"
            )
        if not ways:
            raise SimplydataError(f".{step}() needs to know how to work out the column. Choose one way:\n{ADD_COLUMN_WAYS}")
        if len(ways) > 1:
            raise SimplydataError(
                f".{step}() was given {len(ways)} ways at once ({', '.join(ways)}). Choose exactly one:\n{ADD_COLUMN_WAYS}"
            )
        way = ways[0]
        if way != "from_column=" and given_numbers:
            raise SimplydataError(f"{given_numbers[0]}= only works together with from_column=.\n{FROM_COLUMN_EXAMPLE}")
        if separator is not None and way != "by_joining_text=":
            raise SimplydataError(
                "separator= only works together with by_joining_text=, for example\n"
                'table.add_column("name_and_class", by_joining_text=["name", "class"], separator=" ")'
                '   (with sdata.example("marks"))'
            )
        if for_each is not None and way != "a formula":
            raise SimplydataError(
                "for_each= only works with a formula that has a summary word in it, for example\n"
                'table.add_column("share", "marks / total(marks) * 100", for_each="class")   (with sdata.example("marks"))'
            )

        if way == "a formula":
            share_example = ('For example: table.add_column("share", "marks / total(marks) * 100", for_each="class")'
                             '   (with sdata.example("marks"))')
            reader = _FormulaReader(formula, self._columns)
            expr = reader.read()
            self._need_numbers(*reader.number_used, step=step)
            keys = self._group_keys(for_each, share_example)
            if keys and not reader.summaries:
                raise SimplydataError(
                    "for_each= only matters when the formula has a summary word in it, such as total(marks) or "
                    f"average(marks): then the summary is worked out for each group.\n{share_example}"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                frame, temporary = _apply_summaries(self._frame, reader, keys)
                frame = frame.with_columns(expr.alias(name))
            if temporary:
                frame = frame.drop(temporary)
            words = formula + (f" (summaries for each {', '.join(map(str, keys))})" if keys else "")
            return frame, words

        if way == "value=":
            if not isinstance(value, (str, int, float, bool)):
                raise SimplydataError(
                    f"value= should be one number or one piece of text, not {type(value).__name__}.\n"
                    'For example: table.add_column("school", value="Hill School")'
                )
            return self._frame.with_columns(nw.lit(value).alias(name)), f"{_plain(value)} in every row"

        if way == "by_running=":
            return self._run_function(step, name, by_running)

        if way == "from_column=":
            if not isinstance(from_column, str):
                raise SimplydataError(f"from_column= should be one column name.\n{FROM_COLUMN_EXAMPLE}")
            self._need_columns(from_column)
            self._need_numbers(from_column, step=step)
            if len(given_numbers) != 1:
                raise SimplydataError(
                    f"from_column= needs exactly one of plus=, minus=, times= or divide_by=.\n{FROM_COLUMN_EXAMPLE}"
                )
            option = given_numbers[0]
            amount = numbers_given[option]
            other = self._number_or_column(amount, option)
            col = nw.col(from_column)
            if option == "plus":
                expr, sign = col + other, "+"
            elif option == "minus":
                expr, sign = col - other, "-"
            elif option == "times":
                expr, sign = col * other, "*"
            else:
                expr, sign = _divide(col, other), "/"
            return self._frame.with_columns(expr.alias(name)), f"{from_column} {sign} {amount}"

        # by_adding= / by_subtracting= / by_multiplying= / by_dividing= / by_joining_text=
        option = way.rstrip("=")
        cols = by_joining_text if option == "by_joining_text" else lists[option]
        if not isinstance(cols, (list, tuple)) or len(cols) < 2 or not all(isinstance(c, str) for c in cols):
            raise SimplydataError(
                f"{option}= needs a list of two or more column names, for example\n"
                f'table.add_column("new", {option}=["hours", "marks"])   (with sdata.example("marks"))'
            )
        self._need_columns(*cols)
        if option == "by_joining_text":
            return self._join_text(name, cols, separator)
        if any(self._kind(c) == "date" for c in cols):
            return self._days_between(name, option, cols)
        self._need_numbers(*cols, step=step)
        expr = nw.col(cols[0])
        for c in cols[1:]:
            if option == "by_adding":
                expr = expr + nw.col(c)
            elif option == "by_subtracting":
                expr = expr - nw.col(c)
            elif option == "by_multiplying":
                expr = expr * nw.col(c)
            else:
                expr = _divide(expr, nw.col(c))
        sign = {"by_adding": " + ", "by_subtracting": " - ", "by_multiplying": " * ", "by_dividing": " / "}[option]
        return self._frame.with_columns(expr.alias(name)), sign.join(cols)

    def _days_between(self, name, option, cols):
        """end - start for two date columns: the number of whole days."""
        example = 'For example: table.add_column("days", by_subtracting=["end", "start"])'
        not_dates = [c for c in cols if self._kind(c) != "date"]
        if option != "by_subtracting" or len(cols) != 2 or not_dates:
            problem = (f"{', '.join(not_dates)} {'is' if len(not_dates) == 1 else 'are'} not dates"
                       if not_dates and option == "by_subtracting" and len(cols) == 2 else
                       f"{option}= with dates only works as by_subtracting= with two date columns")
            raise SimplydataError(f"For the days between two dates, give two date columns: the later one first.\n"
                                  f"Here {problem}.\n{example}")
        end, start = cols
        a, b = nw.col(end), nw.col(start)
        types = self._frame.schema[end], self._frame.schema[start]
        if types[0] == nw.Date and types[1] != nw.Date:
            a = a.cast(types[1])
        elif types[1] == nw.Date and types[0] != nw.Date:
            b = b.cast(types[0])
        seconds = (a - b).dt.total_seconds()
        # whole days, counted towards zero: 1 day 23 hours is 1 day, and -1 day 23 hours is -1 day
        days = nw.when(seconds >= 0).then(seconds // 86400).otherwise(-((-seconds) // 86400))
        try:
            frame = self._frame.with_columns(days.alias(name))
        except Exception as error:  # for example one column with a time zone and one without
            raise SimplydataError(f"The days between {end} and {start} could not be worked out: {error}") from None
        if frame[name].null_count() == 0:
            frame = frame.with_columns(nw.col(name).cast(nw.Int64))
        print(f'"{name}" is the number of whole days from {start} to {end} (negative when {end} is earlier).')
        return frame, f"{end} - {start} (whole days)"

    def _join_text(self, name, cols, separator):
        sep = " " if separator is None else separator
        if not isinstance(sep, str):
            raise SimplydataError('separator= should be text in quotes, like separator=" " or separator=", ".')
        parts = []
        for c in cols:
            kind, col = self._kind(c), nw.col(c)
            if kind == "date":
                text = col.dt.to_string("%Y-%m-%d")
            elif kind == "true/false":
                text = nw.when(~col.is_null()).then(nw.when(col).then(nw.lit("True")).otherwise(nw.lit("False")))
            else:
                text = col.cast(nw.String)
            parts.append(_blank_to_empty(text))
        joined = _blank_to_empty(nw.concat_str(parts, separator=sep, ignore_nulls=True))
        return self._frame.with_columns(joined.alias(name)), f"{', '.join(cols)} joined with {_plain(sep)}"

    def _run_function(self, step, name, function):
        """by_running=: call the user's own function once for each row."""
        example = ('For example:\n'
                   '  def per_hour(row):\n'
                   '      return row["marks"] / row["hours"]\n'
                   f'  table.{step}("{name}", by_running=per_hour)   (with sdata.example("marks"))')
        if isinstance(function, str) or not callable(function):
            raise SimplydataError(
                "by_running= needs the function itself: its name without quotes and without brackets.\n" + example
            )
        fname = getattr(function, "__name__", "your function")
        fname = "your function" if fname == "<lambda>" else f"{fname}()"
        values = []
        for i, row in enumerate(self._frame.iter_rows(named=True)):
            row = {k: _plain_cell(v) for k, v in row.items()}
            try:
                result = function(row)
            except Exception as error:
                shown = ", ".join(f"{k}={_plain(v)}" for k, v in list(row.items())[:8])
                raise SimplydataError(
                    f"{fname} stopped at row {i + 1} ({shown}):\n  {type(error).__name__}: {error}\n"
                    "An empty cell reaches the function as None. A function can check for it:\n"
                    '  if row["hours"] is None: return None'
                ) from None
            values.append(_plain_cell(result))
        first = {}
        for i, v in enumerate(values):
            kind = _value_kind(v)
            if kind is not None:
                first.setdefault(kind, i)
        if "other" in first:
            i = first["other"]
            raise SimplydataError(
                f"{fname} should give back one number, text, True/False or date for each row, "
                f"but for row {i + 1} it gave back a {type(values[i]).__name__}.\n{example}"
            )
        if len(first) > 1:
            mix = ", ".join(f"{kind} (row {i + 1})" for kind, i in first.items())
            raise SimplydataError(f"{fname} gave back a mix of kinds: {mix}. A column holds one kind of value.")
        values = [None if _value_kind(v) is None else v for v in values]
        if "number" in first and any(isinstance(v, float) for v in values):
            values = [None if v is None else float(v) for v in values]
        try:
            series = nw.new_series(name, values, backend=self._frame.implementation)
            frame = self._frame.with_columns(series)
        except Exception as error:
            raise SimplydataError(f"The values from {fname} could not become a column: {error}") from None
        return frame, f"{fname.replace('()', '')}(row) for each row"

    @friendly
    def add_column(
        self,
        name=None,
        formula=None,
        by_adding=None,
        by_subtracting=None,
        by_multiplying=None,
        by_dividing=None,
        by_joining_text=None,
        separator=None,
        by_running=None,
        for_each=None,
        from_column=None,
        plus=None,
        minus=None,
        times=None,
        divide_by=None,
        value=None,
    ):
        """Add a new column, worked out from other columns in plain words or with a formula.

        Choose one way:
          table.add_column("study_and_marks", by_adding=["hours", "marks"])
          table.add_column("half", from_column="marks", divide_by=2)
          table.add_column("school", value="Hill School")         same value in every row
          table.add_column("marks_per_hour", "marks / hours")     calculator style
          table.add_column("name_and_class", by_joining_text=["name", "class"], separator=" ")
        (all with sdata.example("marks")). by_subtracting=["a", "b"] means a - b and
        by_dividing=["a", "b"] means a / b, in the order written. Dividing by zero gives an
        empty cell. In a formula, a column name with spaces goes in backticks: "`max temp` - 2".

        Days between two dates: by_subtracting=["end", "start"] with two date columns gives
        the number of whole days from start to end (a part of a day is not counted).

        Summary words in a formula, average() median() total() smallest() biggest() count(),
        give one value for the whole column, or for each group with for_each=:
          table.add_column("share", "marks / total(marks) * 100", for_each="class")
        gives each mark as a percent of its class total (with sdata.example("marks")).

        by_running= runs a function you wrote once for each row. This is ordinary Python:
        the function gets the row (row["marks"] is that row's mark; an empty cell is None)
        and gives back the new value.
          def per_hour(row):
              return row["marks"] / row["hours"]
          table.add_column("per_hour", by_running=per_hour)   (with sdata.example("marks"))
        """
        self._check_new_name(name, "add_column")
        frame, words = self._work_out(
            "add_column", name, formula, by_adding, by_subtracting, by_multiplying, by_dividing, by_joining_text,
            separator, by_running, for_each, from_column, plus, minus, times, divide_by, value)
        return self._finish(name, frame, f'add column "{name}" = {words}')

    @friendly
    def change_column(
        self,
        column=None,
        formula=None,
        by_adding=None,
        by_subtracting=None,
        by_multiplying=None,
        by_dividing=None,
        by_joining_text=None,
        separator=None,
        by_running=None,
        for_each=None,
        from_column=None,
        plus=None,
        minus=None,
        times=None,
        divide_by=None,
        value=None,
    ):
        """Change the values of a column in place, with the same ways as add_column().

        The column keeps its name and its place in the table; undo() brings the old values back.
          table.change_column("marks", "marks / 10")              (with sdata.example("marks"))
          table.change_column("marks", from_column="marks", plus=5)
          table.change_column("marks", "marks - average(marks)", for_each="class")
        """
        example = 'For example: table.change_column("marks", "marks / 10")   (with sdata.example("marks"))'
        if not isinstance(column, str):
            raise SimplydataError(f".change_column() needs the name of the column to change first.\n{example}")
        self._need_columns(column)
        frame, words = self._work_out(
            "change_column", column, formula, by_adding, by_subtracting, by_multiplying, by_dividing,
            by_joining_text, separator, by_running, for_each, from_column, plus, minus, times, divide_by, value)
        return self._finish(column, frame, f'change column "{column}" = {words}')

    @friendly
    def add_group_column(self, name=None, from_column=None, cut_at=None, names=None, rules=None, otherwise=None):
        """Add a column that puts each row into a group, like turning marks into grades.

        Two ways. With numbers: cut_at= says where one group ends and the next begins; names=
        has one more name than cut_at= has numbers. A value exactly on a cut goes into the
        higher group: with cut_at=[40, 60, 80], a mark of 60 is a "B". Empty cells stay empty.
          table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])
        With rules: each rule is a condition and the group name for rows that meet it. The
        first rule a row meets wins; rows that meet no rule get otherwise= (or stay empty).
        An empty cell never meets a rule.
          table.add_group_column("result", rules={"marks >= 80": "top", "marks >= 50": "pass"}, otherwise="fail")
        (both with sdata.example("marks")). Rules can also be a list of pairs:
        rules=[("marks >= 80", "top"), ("marks >= 50", "pass")].
        """
        step = "add_group_column"
        example = ('For example: table.add_group_column("grade", from_column="marks", '
                   'cut_at=[40, 60, 80], names=["D", "C", "B", "A"])   (with sdata.example("marks"))')
        self._check_new_name(name, step)
        if rules is not None:
            if from_column is not None or cut_at is not None or names is not None:
                raise SimplydataError(
                    "add_group_column() takes rules= or from_column= with cut_at= and names=, not both. Choose one way."
                )
            return self._group_by_rules(name, rules, otherwise)
        if otherwise is not None:
            raise SimplydataError(
                "otherwise= goes with rules=, for example\n"
                'table.add_group_column("result", rules={"marks >= 60": "pass"}, otherwise="fail")'
                '   (with sdata.example("marks"))'
            )
        if not isinstance(from_column, str):
            raise SimplydataError(f"add_group_column() needs from_column= with one column name (or rules=).\n{example}")
        self._need_columns(from_column)
        self._need_numbers(from_column, step=step)
        if _is_number(cut_at):
            cut_at = [cut_at]
        if not isinstance(cut_at, (list, tuple)) or not cut_at or not all(_is_number(c) for c in cut_at):
            raise SimplydataError(f"cut_at= should be a list of numbers.\n{example}")
        if list(cut_at) != sorted(cut_at) or len(set(cut_at)) != len(cut_at):
            raise SimplydataError(
                f"cut_at= numbers should go from small to big, with no number twice, like {sorted(set(cut_at))}."
            )
        if not isinstance(names, (list, tuple)) or len(names) != len(cut_at) + 1:
            got = len(names) if isinstance(names, (list, tuple)) else 0
            raise SimplydataError(
                f"names= needs one more name than cut_at= has numbers: {len(cut_at)} cuts make "
                f"{len(cut_at) + 1} groups, but there are {got} names.\n{example}"
            )
        col = nw.col(from_column)
        expr = nw.lit(names[-1])
        for cut, label in reversed(list(zip(cut_at, names[:-1]))):
            expr = nw.when(col < cut).then(nw.lit(label)).otherwise(expr)
        expr = nw.when(~col.is_null()).then(expr)
        return self._put_column(name, expr, f'add group column "{name}" from {from_column} (cut at {list(cut_at)})')

    def _group_by_rules(self, name, rules, otherwise):
        example = ('For example: table.add_group_column("result", rules={"marks >= 80": "top", "marks >= 50": "pass"}, '
                   'otherwise="fail")   (with sdata.example("marks"))')
        if isinstance(rules, dict):
            pairs = list(rules.items())
        elif isinstance(rules, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in rules):
            pairs = [tuple(p) for p in rules]
        else:
            raise SimplydataError(
                "rules= should pair each condition with its group name, in curly brackets or as a list of pairs.\n"
                + example
            )
        if not pairs:
            raise SimplydataError("rules= needs at least one rule.\n" + example)
        labels = [label for _, label in pairs] + ([otherwise] if otherwise is not None else [])
        kinds = {"text" if isinstance(v, str) else "number" if _is_number(v) else "other" for v in labels}
        if "other" in kinds:
            raise SimplydataError("Each group name (and otherwise=) should be text or a number.\n" + example)
        if len(kinds) > 1:
            raise SimplydataError(
                "The group names (and otherwise=) should all be text or all be numbers, so the column holds one kind."
            )
        from .conditions import _ConditionReader, empty_cells_note  # conditions.py builds on this module

        conditions, used = [], []
        for condition, _ in pairs:
            if not isinstance(condition, str):
                raise SimplydataError(f"Each rule should be a condition in quotes, like \"marks >= 80\".\n{example}")
            reader = _ConditionReader(condition, self._frame, where="rules=")
            true, false = reader.read()
            conditions.append((true.fill_null(False), false.fill_null(False)))
            used += [c for c in reader.used if c not in used]
        result = nw.lit(otherwise) if otherwise is not None else None
        for (cond, _), (_, label) in reversed(list(zip(conditions, pairs))):
            result = (nw.when(cond).then(nw.lit(label)) if result is None
                      else nw.when(cond).then(nw.lit(label)).otherwise(result))
        frame = self._frame.with_columns(result.alias(name))
        counts = []
        for label in dict.fromkeys(labels):
            n = int(frame.select((nw.col(name) == label).fill_null(False).cast(nw.Int64).sum()).item() or 0)
            counts.append(f"{_plain(label)}: {_rows_word(n)}")
        unmatched = frame[name].null_count()
        if unmatched:
            counts.append(f"no group (empty): {_rows_word(unmatched)}")
        print(f'Groups in "{name}": ' + ", ".join(counts) + ".")
        # rows where a rule could not be checked (an empty cell) before an earlier rule matched
        met, unknown = nw.lit(False), nw.lit(False)
        for true, false in conditions:
            unknown = unknown | (~met & ~true & ~false)
            met = met | true
        _, note = empty_cells_note(self._frame, nw.lit(False), ~unknown, used, what="rule")
        if note:
            print(note)
        rule_words = "; ".join(f"{cond} -> {_plain(label)}" for cond, label in pairs)
        if otherwise is not None:
            rule_words += f"; otherwise {_plain(otherwise)}"
        return self._finish(name, frame, f'add group column "{name}" by rules ({rule_words})')

    @friendly
    def add_rank_column(self, name=None, from_column=None, biggest_first=True):
        """Add a column with each row's place, where 1 is the best.

        The biggest value gets place 1 (biggest_first=False gives 1 to the smallest). Rows that
        tie share the best place and the next place is skipped, as in sports tables: 1, 2, 2, 4.
        Empty cells get no place.
        table.add_rank_column("place", from_column="marks")   (with sdata.example("marks"))
        """
        self._check_new_name(name, "add_rank_column")
        if not isinstance(from_column, str):
            raise SimplydataError(
                "add_rank_column() needs from_column= with one column name, for example\n"
                'table.add_rank_column("place", from_column="marks")   (with sdata.example("marks"))'
            )
        if not isinstance(biggest_first, bool):
            raise SimplydataError("biggest_first= should be True or False.")
        self._need_columns(from_column)
        expr = nw.col(from_column).rank("min", descending=biggest_first)
        if self._frame[from_column].null_count() == 0:
            expr = expr.cast(nw.Int64)
        order = "biggest first" if biggest_first else "smallest first"
        return self._put_column(name, expr, f'add rank column "{name}" from {from_column} ({order})')

    @friendly
    def add_is_empty_column(self, name=None, from_column=None):
        """Add a true/false column: True where a cell of from_column is empty.

        Add it before filling empty cells, to keep track of which values were filled in.
        table.add_is_empty_column("sales_was_empty", from_column="sales")   (with sdata.example("shop"))
        """
        from .conditions import _empty_in  # what counts as empty: missing, NaN or blank text

        self._check_new_name(name, "add_is_empty_column")
        if not isinstance(from_column, str):
            raise SimplydataError(
                "add_is_empty_column() needs from_column= with one column name, for example\n"
                'table.add_is_empty_column("sales_was_empty", from_column="sales")   (with sdata.example("shop"))'
            )
        self._need_columns(from_column)
        expr = _empty_in(self._frame, from_column).fill_null(True)
        n = int(self._frame.select(expr.cast(nw.Int64).sum()).item() or 0) if len(self._frame) else 0
        print(f'"{name}" is True in {_rows_word(n)} (where {from_column} is empty) and False in the other '
              f"{_rows_word(len(self._frame) - n)}.")
        return self._put_column(name, expr, f'add is-empty column "{name}" from {from_column}')

    # ------------------------------------------------------------------ dates
    @friendly
    def add_date_part_column(self, name=None, from_column=None, part=None):
        """Add a column with one part of each date, such as its year, month or weekday.

        part= can be "year", "quarter" (1 to 4), "month" (1 to 12), "month name" ("March"),
        "week" (the ISO week number, 1 to 53), "day" (day of the month), "weekday" (1 is
        Monday ... 7 is Sunday, as in the ISO 8601 standard), "weekday name" ("Monday"),
        "day of year" (1 to 366), "hour" or "minute". Empty dates give empty cells.
          table.add_date_part_column("weekday", from_column="date", part="weekday name")
        (with sdata.example("shop"): which weekday was each day?)
        """
        step = "add_date_part_column"
        example = ('For example: table.add_date_part_column("weekday", from_column="date", part="weekday name")'
                   '   (with sdata.example("shop"))')
        self._check_new_name(name, step)
        if not isinstance(from_column, str):
            raise SimplydataError(f"add_date_part_column() needs from_column= with the date column.\n{example}")
        self._need_columns(from_column)
        if part is None:
            raise SimplydataError(
                "add_date_part_column() needs part= to say which part of the date, such as part=\"month\".\n"
                "You can use: " + ", ".join(f'"{p}"' for p in DATE_PARTS)
            )
        self._check_choice(step, "part", part, DATE_PARTS)
        kind = self._kind(from_column)
        if kind != "date":
            hint = (f' If it holds dates written as text or years, first write table.make_dates("{from_column}").'
                    if kind in ("text", "number") else "")
            raise SimplydataError(f'"{from_column}" holds {kind} values, not dates.{hint}\n{example}')
        dtype = self._frame.schema[from_column]
        if part in ("hour", "minute") and dtype == nw.Date:
            raise SimplydataError(f'"{from_column}" holds dates without a time of day, so it has no {part}s.')
        col = nw.col(from_column)
        if part == "week":
            return self._finish(name, self._iso_week(name, from_column), f'add date part column "{name}" = week of {from_column}')
        expr = {
            "year": lambda: col.dt.year(),
            "quarter": lambda: (col.dt.month() - 1) // 3 + 1,
            "month": lambda: col.dt.month(),
            "month name": lambda: col.dt.month().replace_strict(
                dict(enumerate(MONTH_NAMES, 1)), return_dtype=nw.String),
            "day": lambda: col.dt.day(),
            "weekday": lambda: col.dt.weekday(),
            "weekday name": lambda: col.dt.weekday().replace_strict(
                dict(enumerate(WEEKDAY_NAMES, 1)), return_dtype=nw.String),
            "day of year": lambda: col.dt.ordinal_day(),
            "hour": lambda: col.dt.hour(),
            "minute": lambda: col.dt.minute(),
        }[part]()
        if part not in ("month name", "weekday name") and self._frame[from_column].null_count() == 0:
            expr = expr.cast(nw.Int64)
        return self._put_column(name, expr, f'add date part column "{name}" = {part} of {from_column}')

    def _iso_week(self, name, column):
        """ISO week numbers (weeks start on Monday; week 1 holds the year's first Thursday)."""
        native = self._frame[column].to_native()
        if self._frame.implementation.is_polars():
            weeks = native.dt.week().rename(name)
        else:
            import numpy as np
            import pandas as pd

            week = native.dt.isocalendar().week
            values = week.to_numpy(dtype="float64", na_value=np.nan)
            weeks = pd.Series(values if week.isna().any() else values.astype("int64"), index=native.index, name=name)
        return self._frame.with_columns(nw.from_native(weeks, series_only=True).alias(name))

    # ------------------------------------------------------------------ text
    @friendly
    def split_column(self, column=None, separator=" ", into=None):
        """Split a text column into several columns at a separator, such as a comma or a space.

        into= names the new columns; they take the place of the old column. The last new
        column keeps the rest of the text when there are more pieces. Spaces at the ends of
        each piece are removed, and missing pieces give empty cells.
          table.split_column("shop", separator=" ", into=["shop_kind", "shop_word"])   (with sdata.example("fruit"))
        """
        step = "split_column"
        example = ('For example: table.split_column("shop", separator=" ", into=["shop_kind", "shop_word"])'
                   '   (with sdata.example("fruit"))')
        if not isinstance(column, str):
            raise SimplydataError(f".split_column() needs the column to split first.\n{example}")
        self._need_columns(column)
        self._text_of(step, column, example)
        if not isinstance(separator, str) or separator == "":
            raise SimplydataError(f'separator= should be the text to split at, like separator="," or separator=" ".\n{example}')
        if not isinstance(into, (list, tuple)) or len(into) < 2 or not all(isinstance(n, str) and n.strip() for n in into):
            raise SimplydataError(f"into= needs a list of two or more names for the new columns.\n{example}")
        if len(set(into)) != len(into):
            raise SimplydataError("into= needs different names for the new columns.")
        taken = [n for n in into if n in self._columns and n != column]
        if taken:
            raise SimplydataError(
                f"The table already has a column called {', '.join(taken)}. Choose other names in into=, "
                "or remove that column first."
            )
        k = len(into)
        native = self._frame[column].to_native()
        if self._frame.implementation.is_polars():
            parts = native.str.splitn(separator, k)
            pieces = [parts.struct.field(f"field_{i}").rename(into[i]) for i in range(k)]
        else:
            import pandas as pd

            split = native.str.split(separator, n=k - 1, expand=True, regex=False)
            pieces = [split[i].rename(into[i]) if i in split.columns
                      else pd.Series(None, index=native.index, dtype=native.dtype, name=into[i]) for i in range(k)]
        frame = self._frame.with_columns([nw.from_native(p, series_only=True).cast(nw.String).alias(n)
                                          for p, n in zip(pieces, into)])
        frame = frame.with_columns([_blank_to_empty(nw.col(n).str.strip_chars()).alias(n) for n in into])
        position = self._columns.index(column)
        others = [c for c in self._columns if c != column]
        frame = frame.select([*others[:position], *into, *others[position:]])
        source = "_simplydata_source_"
        filled_frame = (frame.with_columns(self._frame[column].alias(source))
                        .filter(~_blank_to_empty(nw.col(source).cast(nw.String)).is_null()))
        short = int(filled_frame.select(nw.col(into[-1]).is_null().cast(nw.Int64).sum()).item() or 0) if len(filled_frame) else 0
        longer = int(filled_frame.select(nw.col(into[-1]).str.contains(separator, literal=True).fill_null(False)
                                         .cast(nw.Int64).sum()).item() or 0) if len(filled_frame) else 0
        gone = "" if column in into else f' The column "{column}" is replaced by them.'
        print(f'Split "{column}" at {_plain(separator)} into {", ".join(into)}.{gone}')
        if short:
            print(f"{_cells_word(short)} had fewer than {k} pieces, so {into[-1]} is empty there.")
        if longer:
            print(f"{_cells_word(longer)} had more than {k} pieces; the rest stays in {into[-1]}.")
        return self._change(f"split {column} at {_plain(separator)} into {', '.join(into)}", frame)

    @friendly
    def replace_text(self, column=None, old=None, with_value=None, pattern=None):
        """Replace a piece of text inside the cells of a text column, wherever it appears.

        The old text is plain text and capitals matter. with_value="" removes it.
          table.replace_text("shop", " shop", with_value="")   (with sdata.example("fruit"))
        pattern= takes a regular expression instead of plain text (a pattern language for
        text: \\d means a digit, \\s a space, + "one or more"):
          table.replace_text("shop", pattern=r"\\s+", with_value="_")
        """
        step = "replace_text"
        example = 'For example: table.replace_text("shop", " shop", with_value="")   (with sdata.example("fruit"))'
        if not isinstance(column, str):
            raise SimplydataError(f".replace_text() needs the column first.\n{example}")
        self._need_columns(column)
        text = self._text_of(step, column, example)
        if (old is None) == (pattern is None):
            raise SimplydataError(
                ".replace_text() needs the old text, or pattern= for a regular expression (one of the two).\n" + example
            )
        if with_value is None:
            raise SimplydataError(f'.replace_text() needs with_value= the new text (with_value="" removes it).\n{example}')
        if not isinstance(with_value, str):
            raise SimplydataError(f'with_value= should be text in quotes, like with_value="{with_value}".')
        if old is not None:
            if not isinstance(old, str) or old == "":
                raise SimplydataError(
                    "The old text should be some text in quotes. To change whole values (also numbers) use "
                    'table.replace_values().\n' + example
                )
            found = text.str.contains(old, literal=True)
            new = text.str.replace_all(old, with_value, literal=True)
            what = _plain(old)
        else:
            self._check_pattern(pattern)
            # the new text is used exactly as written, on both engines
            if self._frame.implementation.is_polars():
                value = with_value.replace("$", "$$")
            else:
                value = with_value.replace("\\", "\\\\")
            found = text.str.contains(pattern, literal=False)
            new = text.str.replace_all(pattern, value, literal=False)
            what = f"the pattern {_plain(pattern)}"
        try:
            with warnings.catch_warnings():  # pandas warns about brackets ( ) in a pattern used for finding
                warnings.simplefilter("ignore", UserWarning)
                n = int(self._frame.select(found.fill_null(False).cast(nw.Int64).sum()).item() or 0) if len(self._frame) else 0
            frame = self._frame.with_columns(new.alias(column))
        except Exception as error:
            raise self._pattern_engine_error(pattern, error) from None
        print(f"Replaced {what} in {_cells_word(n)} of {column}.")
        return self._change(f"replace {what} in {column} with {_plain(with_value)}", frame)

    @staticmethod
    def _check_pattern(pattern):
        if not isinstance(pattern, str) or pattern == "":
            raise SimplydataError('pattern= should be a regular expression in quotes, like pattern=r"\\d+".')
        try:
            re.compile(pattern)
        except re.error as error:
            raise SimplydataError(f'pattern= "{pattern}" is not a regular expression simplydata can read: {error}.') from None

    def _pattern_engine_error(self, pattern, error):
        if pattern is not None and self._frame.implementation.is_polars():
            return SimplydataError(
                f'The fast engine (Polars) could not use the pattern "{pattern}". It does not know a few '
                "pattern pieces, such as look-ahead (?=...) and look-behind (?<=...).\n"
                f"Details: {str(error).splitlines()[0]}"
            )
        return SimplydataError(f"The text could not be changed: {str(error).splitlines()[0]}")

    @friendly
    def add_text_part_column(self, name=None, from_column=None, first_characters=None, last_characters=None,
                             characters=None, before=None, after=None, pattern=None):
        """Add a column with one part of each text, such as its first letters or the text before a mark.

        Choose one way:
          first_characters=3   the first 3 characters (letters, digits, spaces and signs)
          last_characters=3    the last 3 characters
          characters=[2, 4]    characters 2 to 4, counting from 1
          before=" "           the text before the first " "
          after=" "            the text after the first " "
          pattern=r"\\d+"       the first piece that fits a regular expression; with
                               brackets ( ) in it, only the part inside the first brackets
        Cells where the mark or pattern is not found give empty cells.
          table.add_text_part_column("initial", from_column="name", first_characters=1)   (with sdata.example("marks"))
        """
        step = "add_text_part_column"
        example = ('For example: table.add_text_part_column("initial", from_column="name", first_characters=1)'
                   '   (with sdata.example("marks"))')
        self._check_new_name(name, step)
        if not isinstance(from_column, str):
            raise SimplydataError(f"add_text_part_column() needs from_column= with the text column.\n{example}")
        self._need_columns(from_column)
        text = self._text_of(step, from_column, example)
        options = {"first_characters": first_characters, "last_characters": last_characters,
                   "characters": characters, "before": before, "after": after, "pattern": pattern}
        given = [k for k, v in options.items() if v is not None]
        if len(given) != 1:
            start = "needs one way to say which part" if not given else f"was given {len(given)} ways at once"
            raise SimplydataError(f"add_text_part_column() {start}. Choose one:\n{TEXT_PART_WAYS}")
        option = given[0]
        value = options[option]
        if option in ("first_characters", "last_characters"):
            if not _whole_number(value) or value < 1:
                raise SimplydataError(f"{option}= should be a whole number of characters, like {option}=3.")
            part = text.str.head(int(value)) if option == "first_characters" else text.str.tail(int(value))
            words = f"{option.split('_')[0]} {value} character{'' if value == 1 else 's'} of {from_column}"
            frame = self._frame.with_columns(_blank_to_empty(part).alias(name))
            return self._finish(name, frame, f'add text part column "{name}" = {words}')
        if option == "characters":
            if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_whole_number(v) for v in value)
                    or value[0] < 1 or value[1] < value[0]):
                raise SimplydataError(
                    "characters= needs two whole numbers: the first and the last character to keep, counting "
                    "from 1. characters=[2, 4] keeps characters 2, 3 and 4."
                )
            part = text.str.slice(int(value[0]) - 1, int(value[1]) - int(value[0]) + 1)
            frame = self._frame.with_columns(_blank_to_empty(part).alias(name))
            words = f"characters {value[0]} to {value[1]} of {from_column}"
            return self._finish(name, frame, f'add text part column "{name}" = {words}')
        if option in ("before", "after"):
            if not isinstance(value, str) or value == "":
                raise SimplydataError(f'{option}= should be the mark in quotes, like {option}="_" or {option}=" ".')
            mark = _escape_mark(value)
            regex = rf"(?s)^(.*?){mark}" if option == "before" else rf"(?s){mark}(.*)$"
            words = f"text {option} the first {_plain(value)} in {from_column}"
            missing = f"no {_plain(value)}"
        else:
            self._check_pattern(pattern)
            regex = pattern if re.compile(pattern).groups else f"({pattern})"
            words = f"the part of {from_column} that fits {_plain(pattern)}"
            missing = "no match for the pattern"
        native = self._frame[from_column].to_native()
        try:
            if self._frame.implementation.is_polars():
                found = native.str.extract(regex, 1)
            else:
                found = native.str.extract(regex, expand=True)[0]
        except Exception as error:
            raise self._pattern_engine_error(pattern, error) from None
        found = nw.from_native(found, series_only=True).cast(nw.String).alias(name)
        frame = self._frame.with_columns(found)
        cleaned = nw.col(name).str.strip_chars() if option in ("before", "after") else nw.col(name)
        frame = frame.with_columns(_blank_to_empty(cleaned).alias(name))
        had_text = ~_blank_to_empty(nw.col(from_column).cast(nw.String)).is_null()
        n = int(frame.select((had_text & nw.col(name).is_null()).cast(nw.Int64).sum()).item() or 0) if len(frame) else 0
        if n:
            print(f'{_cells_word(n)} of {from_column} had {missing}, so "{name}" is empty there.')
        return self._finish(name, frame, f'add text part column "{name}" = {words}')

    # ------------------------------------------------------------------ rolling and previous rows
    def _sequence_checks(self, step, name, from_column, for_each, sorted_by, example, numbers_only=True):
        self._check_new_name(name, step)
        if not isinstance(from_column, str):
            raise SimplydataError(f"{step}() needs from_column= with one column name.\n{example}")
        self._need_columns(from_column)
        if numbers_only:
            self._need_numbers(from_column, step=step)
        return self._group_keys(for_each, example), self._order_columns(sorted_by, example)

    @staticmethod
    def _check_rows_back(rows_back, option="rows_back"):
        if not _whole_number(rows_back) or rows_back < 1:
            raise SimplydataError(f"{option}= should be a whole number of rows, like {option}=1.")
        return int(rows_back)

    @friendly
    def add_rolling_column(self, name=None, from_column=None, window=3, show="average", for_each=None, sorted_by=None):
        """Add a rolling (moving) summary: for each row, the average of this row and the rows just before it.

        window= is how many rows go into each summary (this row and the window - 1 rows
        before it). The first rows of each group are empty until there are enough rows, and a
        window with an empty cell in it gives an empty cell. show= can be "average", "total",
        "median", "smallest", "biggest" or "standard deviation". sorted_by= works it out in the
        order of a column such as a date (the rows themselves keep their place).
          table.add_rolling_column("rain_3_months", from_column="rain", window=3, show="total",
                                   for_each="city", sorted_by="month_number")   (with sdata.example("weather"))
        """
        step = "add_rolling_column"
        example = ('For example: table.add_rolling_column("rain_3_months", from_column="rain", window=3, '
                   'for_each="city", sorted_by="month_number")   (with sdata.example("weather"))')
        keys, order = self._sequence_checks(step, name, from_column, for_each, sorted_by, example)
        if isinstance(window, str):
            raise SimplydataError(
                f'window= counts rows, like window=7. Windows of time such as "{window}" are not ready yet: '
                "sort by the date and use the number of rows."
            )
        if not _whole_number(window) or window < 1:
            raise SimplydataError(f"window= should be a whole number of rows, like window=3.\n{example}")
        window = int(window)
        self._check_choice(step, "show", show, ROLLING_SHOWS)
        if show == "standard deviation" and window < 2:
            raise SimplydataError("A standard deviation needs at least two values, so use window=2 or more.")

        def make(frame, over):
            col = nw.col(from_column)
            if show in ("average", "total", "standard deviation"):
                expr = {"average": col.rolling_mean, "total": col.rolling_sum,
                        "standard deviation": col.rolling_std}[show](window)
                return frame.with_columns(over(expr).alias(name))
            native = frame.to_native()
            if frame.implementation.is_polars():
                import polars as pl

                method = {"median": "rolling_median", "smallest": "rolling_min", "biggest": "rolling_max"}[show]
                expr = getattr(pl.col(from_column), method)(window_size=window)
                native = native.with_columns((expr.over(keys) if keys else expr).alias(name))
            else:
                method = {"median": "median", "smallest": "min", "biggest": "max"}[show]

                def roll(s):
                    return getattr(s.rolling(window), method)()

                if keys:
                    values = native.groupby(keys, dropna=False, sort=False)[from_column].transform(roll)
                else:
                    values = roll(native[from_column])
                native = native.assign(**{name: values})
            return nw.from_native(native, eager_only=True)

        group, how = self._order_words(keys, order)
        before = "this row only" if window == 1 else f"this row and the {_rows_word(window - 1)} before it"
        print(f'Each "{name}" is the {show} of {from_column} in {before}{group}, {how}.')
        return self._row_after_row(
            name, keys, order, make,
            f'add rolling column "{name}" = {show} of {from_column} over {_rows_word(window)}{group}')

    @friendly
    def add_previous_value_column(self, name=None, from_column=None, rows_back=1, for_each=None, sorted_by=None):
        """Add a column with the value from the row before (or rows_back= rows before).

        The first row of each group has no row before it, so it gets an empty cell.
          table.add_previous_value_column("last_month", from_column="temperature", for_each="city",
                                          sorted_by="month_number")   (with sdata.example("weather"))
        """
        step = "add_previous_value_column"
        example = ('For example: table.add_previous_value_column("last_month", from_column="temperature", '
                   'for_each="city", sorted_by="month_number")   (with sdata.example("weather"))')
        keys, order = self._sequence_checks(step, name, from_column, for_each, sorted_by, example, numbers_only=False)
        rows_back = self._check_rows_back(rows_back)

        def make(frame, over):
            return frame.with_columns(over(nw.col(from_column).shift(rows_back)).alias(name))

        group, how = self._order_words(keys, order)
        back = "the row before" if rows_back == 1 else f"{_rows_word(rows_back)} before"
        print(f'Each "{name}" is {from_column} from {back}{group}, {how}.')
        return self._row_after_row(
            name, keys, order, make, f'add previous value column "{name}" = {from_column} from {back}{group}')

    @friendly
    def add_change_column(self, name=None, from_column=None, in_percent=False, rows_back=1, for_each=None,
                          sorted_by=None):
        """Add a column with the change from the row before: this value minus the previous one.

        in_percent=True gives the change as a percent of the previous value,
        (this - previous) / previous * 100; a previous value of 0 gives an empty cell.
        The first row of each group has nothing to compare with, so it is empty.
          table.add_change_column("growth", from_column="population", in_percent=True,
                                  for_each="country", sorted_by="year")   (with sdata.example("growth"))
        """
        step = "add_change_column"
        example = ('For example: table.add_change_column("growth", from_column="population", in_percent=True, '
                   'for_each="country", sorted_by="year")   (with sdata.example("growth"))')
        keys, order = self._sequence_checks(step, name, from_column, for_each, sorted_by, example)
        if not isinstance(in_percent, bool):
            raise SimplydataError("in_percent= should be True or False.")
        rows_back = self._check_rows_back(rows_back)

        def make(frame, over):
            col = nw.col(from_column)
            previous = over(col.shift(rows_back))
            change = col - previous
            expr = _divide(change, previous) * 100 if in_percent else change
            return frame.with_columns(expr.alias(name))

        group, how = self._order_words(keys, order)
        back = "the row before" if rows_back == 1 else f"{_rows_word(rows_back)} before"
        unit = " in percent" if in_percent else ""
        print(f'Each "{name}" is the change in {from_column} from {back}{unit}{group}, {how}.')
        return self._row_after_row(
            name, keys, order, make, f'add change column "{name}" = change in {from_column} from {back}{unit}{group}')

    # ------------------------------------------------------------------ combining tables
    @friendly
    def join_with(self, other=None, using=None, keep="matching rows"):
        """Bring in the columns of another table, matching rows on a column such as a name.

        keep= says which rows stay:
          "matching rows"             only rows found in both tables (the normal choice)
          "all rows from this table"  every row of this table; no match gives empty cells
          "all rows from both"        every row of both tables
        other can be a simplydata table, a pandas or Polars table, or a file name. A column
        found in both tables (other than using=) gets "_other" added to the other table's copy.
        table.join_with({"name": ["Asha", "Ben"], "age": [12, 13]}, using="name")   (with sdata.example("marks"))
        """
        step = "join_with"
        self._check_choice(step, "keep", keep, list(KEEP_CHOICES))
        right = self._other_frame(other, step)
        if using is None:
            shared = [str(c) for c in self._columns if c in right.columns]
            hint = (f"\nColumns in both tables: {', '.join(shared)}" if shared
                    else "\nThe two tables have no column with the same name.")
            raise SimplydataError(
                f'join_with() needs using= to say which column to match rows on, '
                f'for example table.join_with(other, using="name").{hint}'
            )
        keys = _as_list(using)
        self._need_columns(*keys)
        for k in keys:
            if k not in right.columns:
                lines = [f'The other table has no column "{k}".']
                guess = suggest(k, right.columns, n=1)
                if guess:
                    lines.append(f'Did you mean "{guess[0]}"?')
                lines.append("Its columns: " + ", ".join(map(str, right.columns)))
                raise SimplydataError("\n".join(lines))
            left_text, right_text = self._frame.schema[k] == nw.String, right.schema[k] == nw.String
            if left_text != right_text:
                raise SimplydataError(
                    f'Column "{k}" holds {"text" if left_text else "numbers"} in this table but '
                    f'{"text" if right_text else "numbers"} in the other table, so no rows can match.'
                )
        how = KEEP_CHOICES[keep]
        right_keys = right.select(keys).unique()
        try:
            matched = len(self._frame.join(right_keys, on=keys, how="semi"))
            frame = self._frame.join(right, on=keys, how=how, suffix=JOIN_SUFFIX)
        except Exception as e:  # for example whole numbers in one table and decimals in the other
            raise SimplydataError(f"The tables could not be joined on {', '.join(keys)}: {e}") from None
        if how == "full":
            frame = frame.with_columns(
                [nw.coalesce(nw.col(k), nw.col(k + JOIN_SUFFIX)).alias(k) for k in keys]
            ).drop([k + JOIN_SUFFIX for k in keys])
        print(f"{matched} of {len(self._frame)} rows in this table found a match in the other table.")
        if len(right_keys) < len(right):
            print(f"Note: some {', '.join(keys)} values appear more than once in the other table, "
                  "so some rows of this table now appear more than once.")
        return self._change(f"join with another table using {', '.join(keys)} (keep {keep})", frame)

    @friendly
    def add_rows_from(self, other=None):
        """Put the rows of another table under this one; columns are matched by their names.

        table.add_rows_from(sdata.example("marks"))   (with sdata.example("marks"): the class twice)
        """
        bottom = self._other_frame(other, "add_rows_from")
        only_here = [c for c in self._columns if c not in bottom.columns]
        only_there = [c for c in bottom.columns if c not in self._columns]
        if only_here or only_there:
            lines = ["The two tables need the same column names to stack their rows."]
            if only_here:
                lines.append("Only in this table: " + ", ".join(map(str, only_here)))
            if only_there:
                lines.append("Only in the other table: " + ", ".join(map(str, only_there)))
            lines.append("Give the columns the same names first.")
            raise SimplydataError("\n".join(lines))
        bottom = bottom.select(self._columns)
        try:
            if self._frame.implementation.is_polars():
                import polars as pl

                native = pl.concat([self._frame.to_native(), bottom.to_native()], how="vertical_relaxed")
            else:
                import pandas as pd

                native = pd.concat([self._frame.to_native(), bottom.to_native()], ignore_index=True)
        except Exception:
            different = [str(c) for c in self._columns if self._frame.schema[c] != bottom.schema[c]]
            raise SimplydataError(
                "The rows could not be stacked because some columns hold different kinds of values "
                "(for example numbers in one table and text in the other): " + ", ".join(different)
            ) from None
        return self._change(f"add {len(bottom)} rows from another table", nw.from_native(native, eager_only=True))

    # ------------------------------------------------------------------ reshaping
    @friendly
    def turn_columns_into_rows(self, columns=None, names_to="column", values_to="value"):
        """Turn some columns into rows: each value gets its own row, with its old column name beside it.

        This makes a "long" table, which most charts like. The new column names_to= holds the
        old column names and values_to= holds the values.
        table.turn_columns_into_rows(["temperature", "rain"])   (with sdata.example("weather"))
        """
        example = 'For example: table.turn_columns_into_rows(["temperature", "rain"])   (with sdata.example("weather"))'
        cols = _as_list(columns) if columns is not None else []
        if not cols or not all(isinstance(c, str) for c in cols):
            raise SimplydataError(f"turn_columns_into_rows() needs the columns to turn into rows.\n{example}")
        self._need_columns(*cols)
        for option, value, normal in (("names_to", names_to, "column"), ("values_to", values_to, "value")):
            if not isinstance(value, str) or not value:
                raise SimplydataError(f'{option}= should be a name for the new column, like {option}="{normal}".')
        if names_to == values_to:
            raise SimplydataError("names_to= and values_to= need different names.")
        index = [c for c in self._columns if c not in cols]
        for option, value in (("names_to", names_to), ("values_to", values_to)):
            if value in index:
                raise SimplydataError(
                    f'The table already has a column "{value}". Choose another name, '
                    f'for example {option}="{value}_2".'
                )
        text = [c for c in cols if self._frame.schema[c] == nw.String]
        if text and len(text) < len(cols):
            raise SimplydataError(
                "The columns should all hold numbers or all hold text. These hold text: " + ", ".join(text)
            )
        try:
            frame = self._frame.unpivot(on=cols, index=index or None, variable_name=names_to, value_name=values_to)
        except Exception as e:
            raise SimplydataError(f"Those columns could not be turned into rows: {e}") from None
        return self._change(f"turn columns {', '.join(cols)} into rows", frame)

    @friendly
    def turn_rows_into_columns(self, names_from=None, values_from=None):
        """Turn rows into columns: each different value in names_from= becomes a new column, filled from values_from=.

        This makes a "wide" table, the opposite of turn_columns_into_rows(). All the other
        columns say which row is which, so rows only merge when those all agree.
        table.turn_rows_into_columns(names_from="city", values_from="temperature")   (with sdata.example("weather"))
        """
        example = ('For example: table.turn_rows_into_columns(names_from="city", values_from="temperature")'
                   '   (with sdata.example("weather"))')
        if not isinstance(names_from, str) or not isinstance(values_from, str):
            raise SimplydataError(
                f"turn_rows_into_columns() needs names_from= and values_from=, each one column name.\n{example}"
            )
        self._need_columns(names_from, values_from)
        if names_from == values_from:
            raise SimplydataError("names_from= and values_from= need two different columns.")
        index = [c for c in self._columns if c not in (names_from, values_from)]
        if not index:
            raise SimplydataError("There is no other column left to say which row is which. Keep at least one more column.")
        if self._frame.select([*index, names_from]).is_duplicated().any():
            raise SimplydataError(
                f"Some rows have the same {names_from} and the same {', '.join(map(str, index))}, so simplydata "
                f"does not know which {values_from} to put in the new column.\n"
                "Make one row for each of them first, for example with an average:\n"
                f'  table.average_of("{values_from}", for_each=["{index[0]}", "{names_from}"])\n'
                "or remove the columns you do not need first."
            )
        # keep the rows in the order they first appear (pandas would sort them otherwise)
        order = "_simplydata_order_"
        frame = self._frame.with_row_index(order).with_columns(nw.col(order).min().over(*index))
        frame = frame.pivot(on=names_from, index=[order, *index], values=values_from)
        frame = frame.sort(order).drop(order)
        frame = frame.rename({c: str(c) for c in frame.columns if not isinstance(c, str)})
        if frame.implementation.is_pandas_like():  # remove the leftover name pandas puts on the columns
            frame = nw.from_native(frame.to_native().rename_axis(columns=None), eager_only=True)
        return self._change(f"turn rows into columns ({names_from} becomes columns, filled from {values_from})", frame)
