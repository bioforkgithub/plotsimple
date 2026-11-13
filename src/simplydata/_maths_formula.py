"""The shared maths formula reader: formula text into a NumPy function, read safely.

read_formula("x**2 - 4*x + 1") gives a MathsFormula: call it with values for its names
(numbers or NumPy arrays) and it gives back NumPy numbers. read_equation("x**3 - 2*x = 5")
gives the formula "left minus right" and the sign between the two sides.

The text is read with Python's ast module and never run, by the same reader as
table.fit_formula: names that are columns are data, pi is the constant 3.14159..., and any
other name (x, y, a, rate) is a value to find. The functions are those of fit_formula:
exp log sqrt abs sin cos tan erf erfc gamma lgamma digamma beta besselJ.

Used by sdata.find_smallest, find_biggest and solve, and by the fitting, distribution and
curve words.
"""

from __future__ import annotations

import ast
import re

import numpy as np

from ._errors import SimplydataError
from .table.columns import CONSTANTS, FORMULA_HELP
from .table.stat_tests import FIT_HELP, _FitReader

MATHS_HELP = (
    "A formula can use:\n"
    "  names to find              x, y, a, b, rate (any name that is not a column)\n"
    "  numbers                    2, 100, 0.5;  pi is 3.14159...\n"
    "  + - * / **                 add, subtract, multiply, divide, power (x ** 2, not x ^ 2)\n"
    "  ( )                        brackets\n"
    "  exp(x)  log(x)  log(x, 10) e to the power x, natural log, log to base 10\n"
    "  sqrt(x)  abs(x)            square root, size without the minus sign\n"
    "  sin(x)  cos(x)  tan(x)     sine, cosine, tangent (x in radians)\n"
    "  erf(x)  erfc(x)            error function, and 1 - erf(x)\n"
    "  gamma(x)  lgamma(x)        gamma function, and the log of its size (as R)\n"
    "  digamma(x)  beta(a, b)     digamma function, beta function (as R)\n"
    "  besselJ(x, nu)             Bessel function of the first kind, order nu (as R)\n"
    'Example: sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])'
)
# signs between the two sides of an equation or rule; the first spelling is the one used
SIGNS = {"<=": "<=", "=<": "<=", "≤": "<=", ">=": ">=", "=>": ">=", "≥": ">=", "==": "=", "=": "=",
         "<": "<", ">": ">"}
_SIGN_PATTERN = re.compile(r"<=|>=|=<|=>|==|!=|≤|≥|=|<|>")
MAX_POWER = 100  # whole powers up to this are expanded when looking for a polynomial


class _NoColumns(Exception):
    """Raised inside the reader when a formula has no column (allowed here)."""


def _maths_error(problem: str) -> SimplydataError:
    return SimplydataError(f"{problem}\n\n{MATHS_HELP}")


class _MathsReader(_FitReader):
    """fit_formula's reader, allowed to have no column, with known values and a kept tree."""

    def __init__(self, text, columns, known):
        super().__init__(text, columns)
        self.known = known
        self.known_used: list[str] = []
        self.tree = None
        self.expression = None
        self.compiled: dict = {}

    def read(self):
        try:
            return super().read()
        except _NoColumns:
            return self.expression

    def _top(self, node):
        self.tree = node
        self.expression = self._node(node)
        return self.expression

    def _no_column_error(self):
        return _NoColumns()

    def _node(self, node):
        out = super()._node(node)
        self.compiled[id(node)] = out
        return out

    def _value_name(self, name):
        if name in self.known:
            if name not in self.known_used:
                self.known_used.append(name)
            value = self.known[name]
            return lambda p, d: value
        return super()._value_name(name)


def _known_values(known) -> dict:
    """known= as a dict of name -> number. A one-row result table is read by its columns."""
    if known is None:
        return {}
    if hasattr(known, "_frame"):
        frame = known._frame
        if len(frame) != 1:
            raise SimplydataError(
                f"known= takes a one-row result table (such as the one table.fit_formula() gives back); this table "
                f"has {len(frame)} rows. Pick one row first, or give a dict such as known={{\"a\": 2.1}}."
            )
        found = {}
        for column in frame.columns:
            value = frame[column].to_list()[0]
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                if np.isfinite(value):
                    found[str(column)] = float(value)
        return found
    if not isinstance(known, dict):
        raise SimplydataError('known= should look like {"a": 2.1, "b": 0.3}, or be a one-row result table '
                              "from table.fit_formula().")
    found = {}
    for name, value in known.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise SimplydataError(f'known= needs a number for each name; "{name}" has {value!r}. '
                                  'For example known={"a": 2.1}.')
        found[str(name)] = float(value)
    return found


def _free_names(node, free) -> set:
    """The names to find inside a piece of the formula (function names are not counted)."""
    found = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, ast.Call):
            stack.extend(item.args)
            continue
        if isinstance(item, ast.Name):
            if item.id in free:
                found.add(item.id)
            continue
        stack.extend(ast.iter_child_nodes(item))
    return found


class MathsFormula:
    """A formula read safely, ready to work out with NumPy: formula(values, data).

    names         the names to find (variables), in order of first use
    columns_used  the columns the formula uses (empty for the sdata. words)
    known         the names given a fixed value with known=
    text          the formula as written
    """

    def __init__(self, text, function, names, columns_used, known, reader):
        self.text = text
        self._function = function
        self.names = list(names)
        self.columns_used = list(columns_used)
        self.known = dict(known)
        self._reader = reader
        self._polynomial = None
        self._polynomial_done = False

    def __repr__(self):
        return f"MathsFormula({self.text!r}, names={self.names})"

    def __call__(self, values=None, data=None):
        values = {} if values is None else values
        if not isinstance(values, dict):
            if len(self.names) == 1:
                values = {self.names[0]: values}
            else:
                example = ", ".join(repr(n) + ": 1" for n in self.names)
                raise SimplydataError(f"Give the values as a dict, for example {{{example}}}.")
        missing = [n for n in self.names if n not in values]
        if missing:
            raise SimplydataError(f"The formula {self.text} needs a value for {', '.join(missing)}.")
        data = {} if data is None else data
        absent = [c for c in self.columns_used if c not in data]
        if absent:
            raise SimplydataError(f"The formula {self.text} needs the column {', '.join(absent)}.")
        with np.errstate(all="ignore"):
            out = self._function(values, data)
        return np.asarray(out, dtype=float)

    # ------------------------------------------------------------ structure
    def _poly(self, node):
        """The formula as a polynomial in the names: {exponents: coefficient}, or None if it is not one."""
        names = self.names
        zero = (0,) * len(names)
        if isinstance(node, ast.Constant):
            return {zero: float(node.value)}
        if isinstance(node, ast.Name):
            if node.id in names:
                powers = [0] * len(names)
                powers[names.index(node.id)] = 1
                return {tuple(powers): 1.0}
            if node.id in self._reader.placeholders or node.id in self._reader.columns:
                return None
            if node.id in self.known:
                return {zero: float(self.known[node.id])}
            if node.id in CONSTANTS:
                return {zero: float(CONSTANTS[node.id])}
            return None
        if isinstance(node, ast.UnaryOp):
            inner = self._poly(node.operand)
            if inner is None:
                return None
            return {k: -v for k, v in inner.items()} if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            left, right = self._poly(node.left), self._poly(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
                out = dict(left)
                for k, v in right.items():
                    out[k] = out.get(k, 0.0) + sign * v
                return out
            if isinstance(node.op, ast.Mult):
                return _times(left, right)
            right_constant = set(right) <= {zero}
            if isinstance(node.op, ast.Div):
                if not right_constant or right.get(zero, 0.0) == 0:
                    return None
                return {k: v / right[zero] for k, v in left.items()}
            if isinstance(node.op, ast.Pow) and right_constant:
                power = right.get(zero, 0.0)
                if set(left) <= {zero}:
                    with np.errstate(all="ignore"):
                        return {zero: float(np.power(left.get(zero, 0.0), power))}
                if float(power).is_integer() and 0 <= power <= MAX_POWER:
                    out = {zero: 1.0}
                    for _ in range(int(power)):
                        out = _times(out, left)
                    return out
            return None
        if isinstance(node, ast.Call) and not _free_names(node, set(self.names)):
            if self.columns_used and any(isinstance(n, ast.Name) and (n.id in self._reader.columns or
                                         n.id in self._reader.placeholders) for n in ast.walk(node)):
                return None
            compiled = self._reader.compiled.get(id(node))
            if compiled is None:
                return None
            with np.errstate(all="ignore"):
                value = float(np.asarray(compiled({}, {}), dtype=float))
            return {zero: value}
        return None

    def _polynomial_terms(self):
        if not self._polynomial_done:
            terms = self._poly(self._reader.tree)
            if terms is not None:
                terms = {k: v for k, v in terms.items() if v != 0}
                if not all(np.isfinite(v) for v in terms.values()):
                    terms = None
            self._polynomial = terms
            self._polynomial_done = True
        return self._polynomial

    def linear_parts(self):
        """(dict name -> coefficient, constant) when the formula is a straight-line (linear) one, else None."""
        terms = self._polynomial_terms()
        if terms is None or any(sum(k) > 1 for k in terms):
            return None
        coefficients = {name: 0.0 for name in self.names}
        constant = 0.0
        for powers, value in terms.items():
            if sum(powers) == 0:
                constant += value
            else:
                coefficients[self.names[powers.index(1)]] += value
        return coefficients, constant

    def polynomial(self):
        """The coefficients (constant first, as R polyroot()) when the formula is a polynomial in one name, else None."""
        terms = self._polynomial_terms()
        if terms is None or len(self.names) != 1:
            return None
        degree = max((k[0] for k in terms), default=0)
        coefficients = [0.0] * (degree + 1)
        for (power,), value in terms.items():
            coefficients[power] = value
        return coefficients


def _times(left: dict, right: dict) -> dict:
    out: dict = {}
    for k1, v1 in left.items():
        for k2, v2 in right.items():
            key = tuple(a + b for a, b in zip(k1, k2))
            out[key] = out.get(key, 0.0) + v1 * v2
    return out


def read_formula(text, columns=(), known=None) -> MathsFormula:
    """Read formula text safely into a MathsFormula.

    read_formula("x**2 - 4*x + 1");  read_formula("a * exp(-b * time)", columns=["time"]);
    known={"a": 2.1} (or a one-row result table from fit_formula) fixes some names.
    """
    if not isinstance(text, str) or not text.strip():
        raise _maths_error('The formula should be some text, for example "x**2 - 4*x + 1".')
    signs = _SIGN_PATTERN.findall(text)
    if signs:
        raise _maths_error(
            f'"{text}" has {signs[0]} in it, so it is an equation or a rule, not a formula. Here give only the '
            'formula, for example "x**2 - 4*x + 1"; to solve an equation use sdata.solve("x**3 - 2*x = 5").'
        )
    values = _known_values(known)
    reader = _MathsReader(text, [c for c in columns], values)
    try:
        function = reader.read()
    except SimplydataError as error:
        message = str(error).replace(FIT_HELP, MATHS_HELP).replace(FORMULA_HELP, MATHS_HELP)
        if not columns:  # fit_formula's examples use a column; here the example is a name to find
            message = message.replace("hours ** 2", "x ** 2").replace("(hours)", "(x)")
        message = message.replace(" in a formula to fit", " in a formula")
        raise SimplydataError(message) from None
    return MathsFormula(text.strip(), function, reader.values_to_find, reader.used,
                        {k: values[k] for k in reader.known_used}, reader)


def split_equation(text):
    """(left text, sign, right text) of an equation or rule such as "x + y <= 4"."""
    if not isinstance(text, str) or not text.strip():
        raise _maths_error('An equation should be some text, for example "x**3 - 2*x = 5".')
    signs = _SIGN_PATTERN.findall(text)
    if not signs:
        raise _maths_error(f'"{text}" has no = sign. An equation needs one, for example "x**3 - 2*x = 5" '
                           '(or <= and >= for a rule, such as "x + y <= 4").')
    if "!=" in signs:
        raise _maths_error(f'"{text}": != (not equal) cannot be an equation or a rule. Use =, <= or >=.')
    if len(signs) > 1:
        raise _maths_error(f'"{text}" has {len(signs)} signs ({" ".join(signs)}). Write one equation or rule at a '
                           'time, with one =, <= or >=, for example ["x >= 0", "x <= 4"].')
    left, right = _SIGN_PATTERN.split(text, maxsplit=1)
    if not left.strip() or not right.strip():
        raise _maths_error(f'"{text}" needs something on both sides of {signs[0]}, for example "x**3 - 2*x = 5".')
    return left.strip(), SIGNS[signs[0]], right.strip()


def read_equation(text, columns=(), known=None):
    """Read an equation or rule: (MathsFormula of left minus right, sign).

    read_equation("x**3 - 2*x = 5") -> (formula of (x**3 - 2*x) - (5), "=");
    "x + y <= 4" -> "<="; also ">=", and "<" / ">" as written (the caller treats them as <=, >=, with a note).
    """
    left, sign, right = split_equation(text)
    formula = read_formula(f"({left}) - ({right})", columns=columns, known=known)
    formula.equation = text.strip()
    formula.left_text, formula.right_text = left, right
    return formula, sign
