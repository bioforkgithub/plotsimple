"""Sorting, new columns, combining tables, reshaping.

Every word works on pandas and Polars tables alike, because the work is done with
narwhals. Only add_rows_from() uses each engine's own code (so that whole numbers and
decimal numbers can be stacked without fuss).

Calculator-style formulas, add_column("total", "maths + english"), are never run as
Python code. The text is read with Python's ast module and only these pieces are allowed:
column names, numbers, + - * / ** // %, brackets, a minus sign in front, and the
functions round, abs, sqrt and log. Everything else is refused with a friendly message.
"""

from __future__ import annotations

import ast
import keyword
import re
import warnings

import narwhals as nw

from .._data import load_native
from .._errors import PlotsimpleError, column_error, friendly, suggest

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
    'Example: table.add_column("marks_per_hour", "marks / hours")   (with ps.example("marks"))'
)

FUNCTIONS = {"round": (1, 2), "abs": (1, 1), "sqrt": (1, 1), "log": (1, 2)}
MAX_FORMULA_LENGTH = 500

KEEP_CHOICES = {
    "matching rows": "inner",
    "all rows from this table": "left",
    "all rows from both": "full",
}
JOIN_SUFFIX = "_other"


def _formula_error(problem: str) -> PlotsimpleError:
    return PlotsimpleError(f"{problem}\n\n{FORMULA_HELP}")


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


class _FormulaReader:
    """Turns formula text into a narwhals expression. The text is never run."""

    def __init__(self, text, columns):
        self.text = text
        self.columns = list(columns)
        self.used: list = []
        self.placeholders: dict = {}

    def read(self):
        text = self.text
        if not isinstance(text, str) or not text.strip():
            raise _formula_error('The formula should be some text, for example "marks / hours".')
        if len(text) > MAX_FORMULA_LENGTH:
            raise _formula_error(f"That formula is very long. Keep it under {MAX_FORMULA_LENGTH} letters.")

        def backtick(match):
            name = match.group(1)
            if name not in self.columns:
                raise column_error(name, self.columns)
            return self._placeholder(name)

        text = re.sub(r"`([^`]*)`", backtick, text)
        if "`" in text:
            raise _formula_error(
                "A backtick ` is missing. Put a column name with spaces between two backticks: `max temp`."
            )
        text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
        # columns named like Python words (for example "class") still work without backticks
        for name in self.columns:
            if isinstance(name, str) and keyword.iskeyword(name):
                text = re.sub(rf"(?<![\w.]){re.escape(name)}(?!\w)", self._placeholder(name), text)
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            raise _formula_error(
                f'plotsimple cannot read the formula "{self.text}". Check the brackets and signs.'
            ) from None
        except (ValueError, RecursionError, MemoryError):
            raise _formula_error(f'plotsimple cannot read the formula "{self.text}".') from None
        expr = self._node(tree.body)
        if not self.used:
            raise _formula_error(
                "The formula has no column in it. For the same value in every row use value=,\n"
                'for example table.add_column("school", value="Hill School").'
            )
        return expr

    def _placeholder(self, name):
        key = f"_plotsimple_column_{len(self.placeholders)}_"
        self.placeholders[key] = name
        return key

    def _snippet(self, node):
        text = ast.unparse(node)
        for key, name in self.placeholders.items():
            text = text.replace(key, f"`{name}`")
        return text

    def _column(self, name):
        if name not in self.used:
            self.used.append(name)
        return nw.col(name)

    def _node(self, node):
        if isinstance(node, ast.Constant):
            if not _is_number(node.value):
                raise _formula_error(
                    f"{node.value!r} is not allowed in a formula: only numbers and column names are."
                )
            return nw.lit(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.placeholders:
                return self._column(self.placeholders[node.id])
            if node.id in self.columns:
                return self._column(node.id)
            if node.id in FUNCTIONS:
                raise _formula_error(f"{node.id} needs brackets after it, for example {node.id}(marks).")
            raise column_error(node.id, self.columns)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self._node(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.BitXor):
                raise _formula_error("For a power use ** instead of ^, for example height ** 2.")
            kinds = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
            if not isinstance(op, kinds):
                raise _formula_error(f'The sign in "{self._snippet(node)}" is not allowed in a formula.')
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
            raise _formula_error(
                f'"{self._snippet(node)}" is not allowed: a formula cannot use a dot after a name.'
            )
        raise _formula_error(f'"{self._snippet(node)}" is not allowed in a formula.')

    def _call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else self._snippet(node.func)
            guess = suggest(name, list(FUNCTIONS), n=1)
            hint = f" Did you mean {guess[0]}()?" if guess else ""
            raise _formula_error(f'"{name}()" is not allowed in a formula.{hint}')
        name = node.func.id
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise _formula_error(f"Write {name}() simply, like {name}(marks).")
        low, high = FUNCTIONS[name]
        if not low <= len(node.args) <= high:
            raise _formula_error(f"{name}() was given {len(node.args)} things inside the brackets.")
        inner = self._node(node.args[0])
        if name == "abs":
            return inner.abs()
        if name == "sqrt":
            return inner.cast(nw.Float64).sqrt()
        second = node.args[1] if len(node.args) == 2 else None
        if second is not None and not (isinstance(second, ast.Constant) and _is_number(second.value)):
            raise _formula_error(
                f"The second thing in {name}() must be a plain number, for example {name}(marks, 1)."
            )
        if name == "round":
            digits = 0 if second is None else second.value
            if not isinstance(digits, int) or digits < 0:
                raise _formula_error("round(x, 1) needs a whole number of decimal places, like 0, 1 or 2.")
            # School rounding: a half goes away from zero (2.5 -> 3, -2.5 -> -3), the way
            # children learn it. Computers normally round halves to the even number (2.5 -> 2).
            # round(..., 9) first removes tiny computer errors such as 2.675 * 100 = 267.4999...
            scale = 10 ** digits
            scaled = (inner.cast(nw.Float64) * scale).round(9)
            up = (scaled.abs() + 0.5) // 1
            return (nw.when(scaled < 0).then(-up).otherwise(up) / scale).round(digits)
        if second is None:  # log
            return inner.cast(nw.Float64).log()
        if second.value <= 0 or second.value == 1:
            raise _formula_error("The base of log() must be a positive number other than 1, for example log(x, 10).")
        return inner.cast(nw.Float64).log(base=float(second.value))


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


ADD_COLUMN_WAYS = (
    '  a formula        table.add_column("marks_per_hour", "marks / hours")\n'
    '  by_adding=       table.add_column("study_and_marks", by_adding=["hours", "marks"])\n'
    "  by_subtracting=, by_multiplying=, by_dividing=   the same way\n"
    '  from_column=     table.add_column("half", from_column="marks", divide_by=2)\n'
    '  value=           table.add_column("school", value="Hill School")\n'
    '(all with ps.example("marks"))'
)
FROM_COLUMN_EXAMPLE = (
    'For example: table.add_column("half", from_column="marks", divide_by=2)   (with ps.example("marks"))'
)


class ColumnWords:
    # ------------------------------------------------------------------ small helpers
    def _need_numbers(self, *names, step="add_column"):
        for name in names:
            dtype = self._frame.schema[name]
            if not dtype.is_numeric():
                numbers = [c for c, d in self._frame.schema.items() if d.is_numeric()]
                kind = "text" if dtype == nw.String else "values that are not numbers"
                raise PlotsimpleError(
                    f'Column "{name}" holds {kind}, so .{step}() cannot do sums with it.\n'
                    "Columns with numbers: " + (", ".join(map(str, numbers)) if numbers else "none")
                )

    @staticmethod
    def _check_new_name(name, step):
        if not isinstance(name, str) or not name.strip():
            raise PlotsimpleError(
                f'.{step}() needs a name for the new column first, for example table.{step}("total", ...).'
            )

    def _put_column(self, name, expr, what):
        """Add (or replace) a column. Impossible answers (like 1/0 or sqrt(-1)) become empty cells."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame = self._frame.with_columns(expr.alias(name))
            if frame.schema[name].is_float():
                frame = frame.with_columns(nw.when(nw.col(name).is_finite()).then(nw.col(name)).alias(name))
        return self._change(what, frame)

    def _number_or_column(self, value, option):
        if isinstance(value, str):
            self._need_columns(value)
            self._need_numbers(value)
            return nw.col(value)
        if not _is_number(value):
            raise PlotsimpleError(f"{option}= should be a number or a column name, not {value!r}.\n{FROM_COLUMN_EXAMPLE}")
        return nw.lit(value)

    def _other_frame(self, other, step):
        """Another table (plotsimple, pandas, Polars or a file name), in this table's engine."""
        if other is None:
            raise PlotsimpleError(f'.{step}() needs another table, for example table.{step}("other_file.xlsx").')
        if getattr(other, "_is_plotsimple_table", False):
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
        table.sort_by("marks", biggest_first=True)   (with ps.example("marks"))
        """
        names = _as_list(columns)
        if not names:
            raise PlotsimpleError(
                'sort_by() needs a column name, for example table.sort_by("marks")   (with ps.example("marks"))'
            )
        if not isinstance(biggest_first, bool):
            raise PlotsimpleError("biggest_first= should be True or False.")
        self._need_columns(*names)
        order_column = "_plotsimple_order_"
        frame = (
            self._frame.with_row_index(order_column)
            .sort([*names, order_column], descending=[biggest_first] * len(names) + [False], nulls_last=True)
            .drop(order_column)
        )
        order = "biggest first" if biggest_first else "smallest first"
        return self._change(f"sort by {', '.join(map(str, names))} ({order})", frame)

    # ------------------------------------------------------------------ new columns
    @friendly
    def add_column(
        self,
        name=None,
        formula=None,
        by_adding=None,
        by_subtracting=None,
        by_multiplying=None,
        by_dividing=None,
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
        (all with ps.example("marks")). by_subtracting=["a", "b"] means a - b and
        by_dividing=["a", "b"] means a / b, in the order written. Dividing by zero gives an
        empty cell. In a formula, a column name with spaces goes in backticks: "`max temp` - 2".
        """
        self._check_new_name(name, "add_column")
        lists = {"by_adding": by_adding, "by_subtracting": by_subtracting,
                 "by_multiplying": by_multiplying, "by_dividing": by_dividing}
        numbers = {"plus": plus, "minus": minus, "times": times, "divide_by": divide_by}
        given_numbers = [k for k, v in numbers.items() if v is not None]
        ways = (["a formula"] if formula is not None else []) + [f"{k}=" for k, v in lists.items() if v is not None]
        ways += (["from_column="] if from_column is not None else []) + (["value="] if value is not None else [])
        if not ways and given_numbers:
            raise PlotsimpleError(
                f"{given_numbers[0]}= needs from_column= to say which column to start from.\n{FROM_COLUMN_EXAMPLE}"
            )
        if not ways:
            raise PlotsimpleError(f".add_column() needs to know how to work out the new column. Choose one way:\n{ADD_COLUMN_WAYS}")
        if len(ways) > 1:
            raise PlotsimpleError(
                f".add_column() was given {len(ways)} ways at once ({', '.join(ways)}). Choose exactly one:\n{ADD_COLUMN_WAYS}"
            )
        way = ways[0]
        if way != "from_column=" and given_numbers:
            raise PlotsimpleError(f"{given_numbers[0]}= only works together with from_column=.\n{FROM_COLUMN_EXAMPLE}")

        if way == "a formula":
            reader = _FormulaReader(formula, self._columns)
            expr = reader.read()
            self._need_numbers(*reader.used)
            return self._put_column(name, expr, f'add column "{name}" = {formula}')

        if way == "value=":
            if not isinstance(value, (str, int, float, bool)):
                raise PlotsimpleError(
                    f"value= should be one number or one piece of text, not {type(value).__name__}.\n"
                    'For example: table.add_column("school", value="Hill School")'
                )
            return self._put_column(name, nw.lit(value), f'add column "{name}" = {_plain(value)} in every row')

        if way == "from_column=":
            if not isinstance(from_column, str):
                raise PlotsimpleError(f"from_column= should be one column name.\n{FROM_COLUMN_EXAMPLE}")
            self._need_columns(from_column)
            self._need_numbers(from_column)
            if len(given_numbers) != 1:
                raise PlotsimpleError(
                    f"from_column= needs exactly one of plus=, minus=, times= or divide_by=.\n{FROM_COLUMN_EXAMPLE}"
                )
            option = given_numbers[0]
            amount = numbers[option]
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
            return self._put_column(name, expr, f'add column "{name}" = {from_column} {sign} {amount}')

        # by_adding= / by_subtracting= / by_multiplying= / by_dividing=
        option = way.rstrip("=")
        cols = lists[option]
        if not isinstance(cols, (list, tuple)) or len(cols) < 2 or not all(isinstance(c, str) for c in cols):
            raise PlotsimpleError(
                f"{option}= needs a list of two or more column names, for example\n"
                f'table.add_column("new", {option}=["hours", "marks"])   (with ps.example("marks"))'
            )
        self._need_columns(*cols)
        self._need_numbers(*cols)
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
        return self._put_column(name, expr, f'add column "{name}" = {sign.join(cols)}')

    @friendly
    def add_group_column(self, name=None, from_column=None, cut_at=None, names=None):
        """Add a column that puts each number into a group, like turning marks into grades.

        cut_at= says where one group ends and the next begins; names= has one more name than
        cut_at= has numbers. A value exactly on a cut goes into the higher group: with
        cut_at=[40, 60, 80], a mark of 60 is a "B". Empty cells stay empty.
        table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])
        (with ps.example("marks"))
        """
        step = "add_group_column"
        example = ('For example: table.add_group_column("grade", from_column="marks", '
                   'cut_at=[40, 60, 80], names=["D", "C", "B", "A"])   (with ps.example("marks"))')
        self._check_new_name(name, step)
        if not isinstance(from_column, str):
            raise PlotsimpleError(f"add_group_column() needs from_column= with one column name.\n{example}")
        self._need_columns(from_column)
        self._need_numbers(from_column, step=step)
        if _is_number(cut_at):
            cut_at = [cut_at]
        if not isinstance(cut_at, (list, tuple)) or not cut_at or not all(_is_number(c) for c in cut_at):
            raise PlotsimpleError(f"cut_at= should be a list of numbers.\n{example}")
        if list(cut_at) != sorted(cut_at) or len(set(cut_at)) != len(cut_at):
            raise PlotsimpleError(
                f"cut_at= numbers should go from small to big, with no number twice, like {sorted(set(cut_at))}."
            )
        if not isinstance(names, (list, tuple)) or len(names) != len(cut_at) + 1:
            got = len(names) if isinstance(names, (list, tuple)) else 0
            raise PlotsimpleError(
                f"names= needs one more name than cut_at= has numbers: {len(cut_at)} cuts make "
                f"{len(cut_at) + 1} groups, but there are {got} names.\n{example}"
            )
        col = nw.col(from_column)
        expr = nw.lit(names[-1])
        for cut, label in reversed(list(zip(cut_at, names[:-1]))):
            expr = nw.when(col < cut).then(nw.lit(label)).otherwise(expr)
        expr = nw.when(~col.is_null()).then(expr)
        return self._put_column(name, expr, f'add group column "{name}" from {from_column} (cut at {list(cut_at)})')

    @friendly
    def add_rank_column(self, name=None, from_column=None, biggest_first=True):
        """Add a column with each row's place, where 1 is the best.

        The biggest value gets place 1 (biggest_first=False gives 1 to the smallest). Rows that
        tie share the best place and the next place is skipped, as in sports tables: 1, 2, 2, 4.
        Empty cells get no place.
        table.add_rank_column("place", from_column="marks")   (with ps.example("marks"))
        """
        self._check_new_name(name, "add_rank_column")
        if not isinstance(from_column, str):
            raise PlotsimpleError(
                "add_rank_column() needs from_column= with one column name, for example\n"
                'table.add_rank_column("place", from_column="marks")   (with ps.example("marks"))'
            )
        if not isinstance(biggest_first, bool):
            raise PlotsimpleError("biggest_first= should be True or False.")
        self._need_columns(from_column)
        expr = nw.col(from_column).rank("min", descending=biggest_first)
        if self._frame[from_column].null_count() == 0:
            expr = expr.cast(nw.Int64)
        order = "biggest first" if biggest_first else "smallest first"
        return self._put_column(name, expr, f'add rank column "{name}" from {from_column} ({order})')

    # ------------------------------------------------------------------ combining tables
    @friendly
    def join_with(self, other=None, using=None, keep="matching rows"):
        """Bring in the columns of another table, matching rows on a column such as a name.

        keep= says which rows stay:
          "matching rows"             only rows found in both tables (the normal choice)
          "all rows from this table"  every row of this table; no match gives empty cells
          "all rows from both"        every row of both tables
        other can be a plotsimple table, a pandas or Polars table, or a file name. A column
        found in both tables (other than using=) gets "_other" added to the other table's copy.
        table.join_with({"name": ["Asha", "Ben"], "age": [12, 13]}, using="name")   (with ps.example("marks"))
        """
        step = "join_with"
        self._check_choice(step, "keep", keep, list(KEEP_CHOICES))
        right = self._other_frame(other, step)
        if using is None:
            shared = [str(c) for c in self._columns if c in right.columns]
            hint = (f"\nColumns in both tables: {', '.join(shared)}" if shared
                    else "\nThe two tables have no column with the same name.")
            raise PlotsimpleError(
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
                raise PlotsimpleError("\n".join(lines))
            left_text, right_text = self._frame.schema[k] == nw.String, right.schema[k] == nw.String
            if left_text != right_text:
                raise PlotsimpleError(
                    f'Column "{k}" holds {"text" if left_text else "numbers"} in this table but '
                    f'{"text" if right_text else "numbers"} in the other table, so no rows can match.'
                )
        how = KEEP_CHOICES[keep]
        right_keys = right.select(keys).unique()
        try:
            matched = len(self._frame.join(right_keys, on=keys, how="semi"))
            frame = self._frame.join(right, on=keys, how=how, suffix=JOIN_SUFFIX)
        except Exception as e:  # for example whole numbers in one table and decimals in the other
            raise PlotsimpleError(f"The tables could not be joined on {', '.join(keys)}: {e}") from None
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

        table.add_rows_from(ps.example("marks"))   (with ps.example("marks"): the class twice)
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
            raise PlotsimpleError("\n".join(lines))
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
            raise PlotsimpleError(
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
        table.turn_columns_into_rows(["temperature", "rain"])   (with ps.example("weather"))
        """
        example = 'For example: table.turn_columns_into_rows(["temperature", "rain"])   (with ps.example("weather"))'
        cols = _as_list(columns) if columns is not None else []
        if not cols or not all(isinstance(c, str) for c in cols):
            raise PlotsimpleError(f"turn_columns_into_rows() needs the columns to turn into rows.\n{example}")
        self._need_columns(*cols)
        for option, value, normal in (("names_to", names_to, "column"), ("values_to", values_to, "value")):
            if not isinstance(value, str) or not value:
                raise PlotsimpleError(f'{option}= should be a name for the new column, like {option}="{normal}".')
        if names_to == values_to:
            raise PlotsimpleError("names_to= and values_to= need different names.")
        index = [c for c in self._columns if c not in cols]
        for option, value in (("names_to", names_to), ("values_to", values_to)):
            if value in index:
                raise PlotsimpleError(
                    f'The table already has a column "{value}". Choose another name, '
                    f'for example {option}="{value}_2".'
                )
        text = [c for c in cols if self._frame.schema[c] == nw.String]
        if text and len(text) < len(cols):
            raise PlotsimpleError(
                "The columns should all hold numbers or all hold text. These hold text: " + ", ".join(text)
            )
        try:
            frame = self._frame.unpivot(on=cols, index=index or None, variable_name=names_to, value_name=values_to)
        except Exception as e:
            raise PlotsimpleError(f"Those columns could not be turned into rows: {e}") from None
        return self._change(f"turn columns {', '.join(cols)} into rows", frame)

    @friendly
    def turn_rows_into_columns(self, names_from=None, values_from=None):
        """Turn rows into columns: each different value in names_from= becomes a new column, filled from values_from=.

        This makes a "wide" table, the opposite of turn_columns_into_rows(). All the other
        columns say which row is which, so rows only merge when those all agree.
        table.turn_rows_into_columns(names_from="city", values_from="temperature")   (with ps.example("weather"))
        """
        example = ('For example: table.turn_rows_into_columns(names_from="city", values_from="temperature")'
                   '   (with ps.example("weather"))')
        if not isinstance(names_from, str) or not isinstance(values_from, str):
            raise PlotsimpleError(
                f"turn_rows_into_columns() needs names_from= and values_from=, each one column name.\n{example}"
            )
        self._need_columns(names_from, values_from)
        if names_from == values_from:
            raise PlotsimpleError("names_from= and values_from= need two different columns.")
        index = [c for c in self._columns if c not in (names_from, values_from)]
        if not index:
            raise PlotsimpleError("There is no other column left to say which row is which. Keep at least one more column.")
        if self._frame.select([*index, names_from]).is_duplicated().any():
            raise PlotsimpleError(
                f"Some rows have the same {names_from} and the same {', '.join(map(str, index))}, so plotsimple "
                f"does not know which {values_from} to put in the new column.\n"
                "Make one row for each of them first, for example with an average:\n"
                f'  table.average_of("{values_from}", for_each=["{index[0]}", "{names_from}"])\n'
                "or remove the columns you do not need first."
            )
        # keep the rows in the order they first appear (pandas would sort them otherwise)
        order = "_plotsimple_order_"
        frame = self._frame.with_row_index(order).with_columns(nw.col(order).min().over(*index))
        frame = frame.pivot(on=names_from, index=[order, *index], values=values_from)
        frame = frame.sort(order).drop(order)
        frame = frame.rename({c: str(c) for c in frame.columns if not isinstance(c, str)})
        if frame.implementation.is_pandas_like():  # remove the leftover name pandas puts on the columns
            frame = nw.from_native(frame.to_native().rename_axis(columns=None), eager_only=True)
        return self._change(f"turn rows into columns ({names_from} becomes columns, filled from {values_from})", frame)
