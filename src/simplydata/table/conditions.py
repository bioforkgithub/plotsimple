"""One reader for written conditions, used everywhere a condition is written as text.

keep_rows("marks > 50 and class == 'A'"), remove_rows(...) and
add_group_column(rules={"marks >= 80": "top"}) all read their conditions here, so they
accept the same things and treat empty cells the same way.

The text is never run as Python code: it is read with Python's ast module and only
comparisons, and/or/not, brackets, "in [...]", "is empty", column names, numbers, text and
dates in quotes and simple sums are allowed.

Empty cells: a comparison with an empty cell is neither true nor false (three-valued logic,
as in SQL). A row only meets a condition when it is surely true, so an empty cell never
meets a condition, and "not" cannot turn an empty cell into a match either. Only
"is empty" finds empty cells.
"""

from __future__ import annotations

import ast
import datetime as _dt
import numbers
import operator
import re

import narwhals as nw

from .._errors import SimplydataError, column_error
from .columns import FORMULA_HELP, _FormulaReader


class _NotGiven:
    def __repr__(self):
        return "not given"


_NOT_GIVEN = _NotGiven()


def _kind(dtype) -> str:
    """The kind of a column in plain words."""
    if dtype == nw.Boolean:
        return "true/false"
    if dtype.is_numeric():
        return "number"
    if dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum:
        return "text"
    if dtype == nw.Date or dtype == nw.Datetime:
        return "date"
    if dtype == nw.Object:
        return "mixed"
    return "other"


def _empty_in(frame, column):
    """True for every empty cell in the column (missing, NaN, or blank text)."""
    dtype = frame.schema[column]
    col = nw.col(column)
    empty = col.is_null()
    if dtype.is_float():
        empty = empty | col.is_nan()
    elif dtype == nw.String:
        empty = empty | (col.str.strip_chars() == "")
    return empty


def _count_in(frame, expr) -> int:
    """How many rows an expression is true for."""
    if len(frame) == 0:
        return 0
    return int(frame.select(expr.fill_null(False).cast(nw.Int64).sum()).item() or 0)


def _say(value) -> str:
    """A value written the way steps() shows it."""
    if isinstance(value, str):
        return f'"{value}"'
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, _dt.datetime):
        if value.hour or value.minute or value.second:
            return value.isoformat(sep=" ")
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if value is None:
        return "empty"
    return str(value)

def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _looks_like_number(text) -> bool:
    try:
        float(str(text).replace(",", "."))
        return True
    except ValueError:
        return False


def _rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


# ---------------------------------------------------------------------------- written conditions

CONDITION_HELP = (
    "A condition can use:\n"
    "  column names            marks, class  (names with spaces go in backticks: `max temp`)\n"
    "  numbers                 50, 2.5\n"
    "  text in quotes          class == 'A'   (capital letters matter)\n"
    "  dates in quotes         date > '2026-03-01'   (year-month-day, for date columns)\n"
    "  ==  !=                  equal to, not equal to\n"
    "  <  <=  >  >=            below, at most, above, at least\n"
    "  and  or  not  ( )       join conditions; brackets decide what goes together\n"
    "  in [...]                subject in ['Maths', 'Science']\n"
    "  is empty                marks is empty, marks is not empty\n"
    "  sums                    marks / hours > 10   (+ - * / ** and round, abs, sqrt, log)\n"
    "An empty cell never meets a condition, and not does not change that: use is empty to find them.\n"
    "Example: table.keep_rows(\"marks > 50 and class == 'A'\")   (with sdata.example(\"marks\"))\n"
    "Example: table.add_group_column(\"result\", rules={\"marks >= 60\": \"pass\"}, otherwise=\"fail\")"
)

_SIGNS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_COMPARE_WITH = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                 ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_FLIPPED = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}


def _condition_error(problem: str) -> SimplydataError:
    return SimplydataError(f"{problem}\n\n{CONDITION_HELP}")


class _Side:
    """One side of a comparison: a column, a sum, or a value written in the condition."""

    def __init__(self, label, kind, expr=None, known=None, column=None, literal=_NOT_GIVEN, has_column=True):
        self.label = label          # how to name it in a message
        self.kind = kind            # number, text, date, true/false, ...
        self.expr = expr            # narwhals expression (not for written values)
        self.known = known          # true where the side is not empty
        self.column = column        # the column name, when the side is just a column
        self.literal = literal      # the value, when it is written in the condition
        self.has_column = has_column

    @property
    def is_value(self) -> bool:
        return self.literal is not _NOT_GIVEN


class _ConditionReader(_FormulaReader):
    """Turns condition text into narwhals expressions. The text is never run.

    read() gives two expressions: where the condition is surely true, and where it is surely
    false. A comparison with an empty cell is neither (three-valued logic, as in SQL), so
    keep_rows keeps only "surely true" rows and "not" cannot bring empty cells back.
    """

    def __init__(self, text, frame, where="keep_rows() or remove_rows()"):
        super().__init__(text, frame.columns)
        self.frame = frame
        self.where = where  # named in messages: "keep_rows() or remove_rows()", "rules="
        self.in_sum = False

    # messages from the formula reader speak of a condition and list what a condition can use
    def _error(self, problem):
        return _condition_error(problem.replace("formula", "condition"))

    def _what(self):
        return "condition"

    def _no_column_error(self):
        return _condition_error("A condition needs a column in it, like marks > 50.")

    def read(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise _condition_error("The condition should be some text, for example \"marks > 50\".")
        try:
            result = super().read()
        except SimplydataError as error:
            message = str(error)
            if CONDITION_HELP in message and "cannot read" in message:
                hint = self._hint()
                if hint:
                    raise _condition_error(message.replace("\n\n" + CONDITION_HELP, "") + f"\n{hint}") from None
            raise self._from_formula(error) from None
        if getattr(self, "summaries", None):  # average(marks) and friends need a column first
            raise _condition_error(
                f"Summary words such as average() cannot be used in {self.where} yet. "
                "Add them as a column first, then compare with it:\n"
                '  table.add_column("class_average", "average(marks)", for_each="class")\n'
                '  table.keep_rows("marks > class_average")'
            )
        return result

    def _from_formula(self, error):
        """A formula-reader message, if one slips through, with the condition help instead."""
        message = str(error)
        if FORMULA_HELP not in message:
            return error
        message = message.replace("\n\n" + FORMULA_HELP, "")
        for old, new in (("the formula", "the condition"), ("a formula", "a condition"),
                         ("That formula", "That condition")):
            message = message.replace(old, new)
        hint = self._hint() if "cannot read" in message else ""
        return _condition_error(message + (f"\n{hint}" if hint else ""))

    def _hint(self) -> str:
        """A guess at what went wrong when the text cannot be read at all."""
        outside = re.sub(r"'[^']*'|\"[^\"]*\"", "''", self.text)  # leave out text in quotes
        if "&&" in outside or "||" in outside:
            return "Write and / or as words: marks > 50 and class == 'A'."
        if "<>" in outside:
            return "For 'not equal to' write != ."
        if re.search(r"(?<![<>=!])=(?!=)", outside):
            return "To compare, write two equal signs: class == 'A'."
        if re.search(r"\b(AND|OR|NOT|And|Or|Not)\b", outside):
            return "Write and, or, not in small letters."
        if self.text.count("'") % 2 or self.text.count('"') % 2:
            return "A quote mark is missing: text needs one at the start and one at the end, like 'A'."
        if outside.count("(") != outside.count(")"):
            return "A bracket is missing: every ( needs a )."
        return ""

    # ---- the whole condition, or one part of it
    def _node(self, node):
        if self.in_sum:  # a sum inside a comparison: the formula reader does it
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                raise _condition_error("Write and / or as words instead of & and |: marks > 50 and class == 'A'.")
            if isinstance(node, (ast.Compare, ast.BoolOp)) or (isinstance(node, ast.UnaryOp)
                                                               and isinstance(node.op, ast.Not)):
                raise _condition_error(f'"{self._snippet(node)}" needs brackets around it.')
            return super()._node(node)
        return self._test(node)

    def _test(self, node):
        """(surely true, surely false) for a condition."""
        if isinstance(node, ast.BoolOp):
            parts = [self._test(value) for value in node.values]
            true, false = parts[0]
            for t, f in parts[1:]:
                if isinstance(node.op, ast.And):
                    true, false = true & t, false | f
                else:
                    true, false = true | t, false & f
            return true, false
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            true, false = self._test(node.operand)
            return false, true
        if isinstance(node, ast.Compare):
            left = self._side(node.left)
            result = None
            for op, right_node in zip(node.ops, node.comparators):  # 40 <= marks <= 60
                part = self._compare(left, op, right_node)
                result = part if result is None else (result[0] & part[0], result[1] | part[1])
                if not isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                    left = self._side(right_node)
            return result
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            raise _condition_error("Write and / or as words instead of & and |: marks > 50 and class == 'A'.")
        if isinstance(node, ast.Name):
            side = self._side(node)
            if side.kind == "true/false":
                col = side.expr
                return col.fill_null(False) & side.known, (~col).fill_null(False) & side.known
            raise _condition_error(
                f'"{side.label}" alone is not a condition. Compare it with something, like {side.label} > 50 '
                f"or {side.label} == 'A'."
            )
        if isinstance(node, ast.Constant):
            raise _condition_error("A condition needs a column in it, like marks > 50.")
        raise _condition_error(
            f'"{self._snippet(node)}" is not a condition. A condition compares, like marks > 50.'
        )

    # ---- one side of a comparison
    def _side(self, node) -> _Side:
        if isinstance(node, ast.Name):
            name = self.placeholders.get(node.id, node.id)
            if name in self.columns:
                self._column(name)
                kind = _kind(self.frame.schema[name])
                return _Side(name, kind, nw.col(name), ~_empty_in(self.frame, name), column=name)
            if node.id == "empty":
                raise _condition_error("To find empty cells write is empty, like: marks is empty.")
            if node.id in ("true", "false"):
                raise _condition_error(f"Write {node.id.title()} with a capital letter and no quotes.")
            raise column_error(node.id, self.columns)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):  # never let a column-name stand-in change quoted text
                for key, name in self.placeholders.items():
                    value = value.replace(key, name)
            return self._written(value)
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant) and _is_number(node.operand.value)):
            value = node.operand.value
            return self._written(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            raise _condition_error("A list in square brackets goes after in, like subject in ['Maths', 'Science'].")
        # a sum, like marks / hours: every column in it must hold numbers
        names = []
        for part in ast.walk(node):
            if isinstance(part, ast.Name):
                name = self.placeholders.get(part.id, part.id)
                if name in self.columns and name not in names:
                    names.append(name)
        for name in names:
            kind = _kind(self.frame.schema[name])
            if kind != "number":
                raise _condition_error(f'"{name}" is a {kind} column, so it cannot be used in a sum.')
        self.in_sum = True
        try:
            expr = self._node(node)
        finally:
            self.in_sum = False
        known = expr.cast(nw.Float64).is_finite().fill_null(False)
        for name in names:
            known = known & ~_empty_in(self.frame, name)
        return _Side(self._snippet(node), "number", expr, known, has_column=bool(names))

    @staticmethod
    def _written(value) -> _Side:
        if value is None:
            raise _condition_error("To find empty cells write is empty, like: marks is empty.")
        if isinstance(value, bool):
            kind = "true/false"
        elif _is_number(value):
            kind = "number"
        elif isinstance(value, str):
            kind = "text"
        else:
            raise _condition_error(f"{value!r} is not allowed in a condition.")
        return _Side(_say(value), kind, literal=value, has_column=False)

    # ---- comparisons
    def _compare(self, left: _Side, op, right_node):
        if isinstance(op, (ast.Is, ast.IsNot)):
            is_empty_word = ((isinstance(right_node, ast.Name) and right_node.id == "empty")
                             or (isinstance(right_node, ast.Constant) and right_node.value is None))
            if not is_empty_word:
                raise _condition_error("After is write empty, like: marks is empty. To compare values use == .")
            if left.column is None:
                raise _condition_error("is empty goes after a column name, like: marks is empty.")
            empty = _empty_in(self.frame, left.column)
            return (empty, ~empty) if isinstance(op, ast.Is) else (~empty, empty)

        if isinstance(op, (ast.In, ast.NotIn)):
            if left.is_value or not isinstance(right_node, (ast.List, ast.Tuple, ast.Set)):
                raise _condition_error("Write in like this: subject in ['Maths', 'Science'].")
            values = []
            for item in right_node.elts:
                side = self._side(item)
                if not side.is_value:
                    raise _condition_error("The list after in can only hold values, like ['Maths', 'Science'].")
                values.append(self._fit(left, side.literal, "=="))
            if not values:
                raise _condition_error("The list after in needs at least one value, like ['Maths', 'Science'].")
            cond = left.expr.is_in(values)
            if isinstance(op, ast.NotIn):
                cond = ~cond
            return cond.fill_null(False) & left.known, (~cond).fill_null(False) & left.known

        if type(op) not in _SIGNS:
            raise _condition_error("That way of comparing is not allowed in a condition.")
        right = self._side(right_node)
        if left.is_value and right.is_value:
            raise _condition_error(
                f"{left.label} and {right.label} are both values. Compare a column with a value, like marks > 50."
            )
        if left.is_value:  # 50 < marks is the same as marks > 50
            left, right, op = right, left, _FLIPPED[type(op)]()
        if not (left.has_column or right.has_column):
            raise _condition_error("A condition needs a column in it, like marks > 50.")
        sign = _SIGNS[type(op)]
        if sign not in ("==", "!=") and left.kind not in ("number", "date"):
            raise _condition_error(
                f'{sign} compares numbers or dates, but "{left.label}" is a {left.kind} column. '
                f"For text use == or !=, or keep_rows(\"{left.label}\", starts_with=\"A\")."
            )
        if right.is_value:
            other, known = self._fit(left, right.literal, sign), left.known
        else:
            if left.kind != right.kind:
                raise _condition_error(
                    f'"{left.label}" is a {left.kind} column and "{right.label}" is a {right.kind} column, '
                    f"so they cannot be compared."
                )
            other, known = right.expr, left.known & right.known
        cond = _COMPARE_WITH[type(op)](left.expr, other)
        return cond.fill_null(False) & known, (~cond).fill_null(False) & known

    def _fit(self, side: _Side, value, sign: str):
        """A written value, checked against the kind of the column it is compared with."""
        label, kind = side.label, side.kind
        if kind == "number":
            if _is_number(value):
                return value
            if isinstance(value, str) and _looks_like_number(value):
                raise _condition_error(
                    f'"{label}" holds numbers, so write the number without quotes: {label} {sign} {value.strip()}'
                )
            raise _condition_error(f'"{label}" holds numbers, so compare it with a number, like {label} {sign} 50.')
        if kind == "text":
            if isinstance(value, str):
                return value
            hint = (f' If "{label}" holds numbers written as text, first use table.make_numbers("{label}").'
                    if _is_number(value) else "")
            raise _condition_error(f"\"{label}\" is a text column, so put the text in quotes: {label} {sign} '{value}'.{hint}")
        if kind == "true/false":
            if isinstance(value, bool):
                return value
            raise _condition_error(f'"{label}" is a true/false column, so compare it with True or False (no quotes).')
        if kind == "date":
            is_datetime = self.frame.schema[side.column] == nw.Datetime if side.column else False
            if isinstance(value, str):
                try:
                    when = _dt.datetime.fromisoformat(value.strip())
                except ValueError:
                    when = None
                if when is not None:
                    return when if is_datetime else when.date()
            raise _condition_error(
                f"\"{label}\" is a date column, so write the date in quotes as year-month-day: {label} {sign} '2026-03-15'."
            )
        raise _condition_error(
            f'"{label}" holds a mix of things. Turn it into text first with table.make_text("{label}").'
        )


def empty_cells_note(frame, true, false, used, what="condition") -> tuple:
    """(rows, note) for the rows a condition could not be checked on because of empty cells.

    Such rows are neither surely true nor surely false. The note names the columns whose
    empty cells caused it; it is "" when no row is affected.
    """
    unknown = ~(true.fill_null(False) | false.fill_null(False))
    n = _count_in(frame, unknown)
    if not n:
        return 0, ""
    names = [c for c in used if c in frame.columns and _count_in(frame, unknown & _empty_in(frame, c))]
    where = ", ".join(map(str, names or used))
    has = "has" if n == 1 else "have"
    return n, f"Note: {_rows(n)} {has} an empty cell in {where}. An empty cell never meets a {what}."
