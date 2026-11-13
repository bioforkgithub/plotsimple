"""Tests for find_smallest, find_biggest, solve and the shared maths formula reader.

Reference values come from R 4.6.2 (base R, lpSolve 5.6, nloptr, nleqslv, rootSolve), with the
R call that made them written next to each value. Where R and SciPy use different algorithms
that land on the same answer (Nelder-Mead in optim() and in SciPy, L-BFGS-B), the place and
value of the answer are compared with a tolerance that fits the method's stopping rule.
"""

import math

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata._maths_formula import MathsFormula, read_equation, read_formula
from simplydata.finding import find_biggest, find_smallest, solve
from simplydata.table import table as make_table


def row(result, index=0):
    frame = result._frame.to_pandas() if hasattr(result._frame, "to_pandas") else result._frame
    return frame.iloc[index].to_dict()


def rows(result):
    return result._frame.to_pandas().to_dict("records")


# ---------------------------------------------------------------------------- the formula reader
def test_read_formula_names_values_and_parts():
    f = read_formula("x**2 - 4*x + 1")
    assert isinstance(f, MathsFormula)
    assert f.names == ["x"] and f.columns_used == []
    assert np.allclose(f({"x": np.array([0.0, 2.0])}), [1.0, -3.0])
    assert f.linear_parts() is None
    assert f.polynomial() == [1.0, -4.0, 1.0]


def test_read_formula_linear_parts_with_constants():
    f = read_formula("3*x + 2*y - sqrt(4)*pi + (x - y)/2")
    coefficients, constant = f.linear_parts()
    assert coefficients == {"x": 3.5, "y": 1.5}
    assert constant == pytest.approx(-2 * math.pi)


def test_read_formula_polynomial_expands_powers():
    assert read_formula("(x + 1)**3 / 2").polynomial() == [0.5, 1.5, 1.5, 0.5]
    assert read_formula("exp(x)").polynomial() is None
    assert read_formula("x * y").polynomial() is None  # two names


def test_read_formula_with_columns_and_known():
    f = read_formula("a * exp(-b * time)", columns=["time"], known={"a": 2})
    assert f.names == ["b"] and f.columns_used == ["time"] and f.known == {"a": 2.0}
    out = f({"b": 1.0}, {"time": np.arange(3.0)})
    assert np.allclose(out, 2 * np.exp(-np.arange(3.0)))
    assert f.linear_parts() is None  # a column makes it not a constant-coefficient formula


def test_read_formula_known_from_a_one_row_table():
    known = make_table(pd.DataFrame({"a": [2.0], "b": [0.5], "rows": [10], "test": ["fit"]}))
    f = read_formula("a * x * exp(-b * x)", known=known)
    assert f.names == ["x"] and f.known == {"a": 2.0, "b": 0.5}
    with pytest.raises(SimplydataError, match="one-row"):
        read_formula("a * x", known=make_table(pd.DataFrame({"a": [1.0, 2.0]})))


def test_read_equation_signs():
    f, sign = read_equation("x + y <= 4")
    assert sign == "<=" and f.linear_parts() == ({"x": 1.0, "y": 1.0}, -4.0)
    f, sign = read_equation("x**3 - 2*x = 5")
    assert sign == "=" and f.polynomial() == [-5.0, -2.0, 0.0, 1.0]
    assert read_equation("x >= 0")[1] == ">=" and read_equation("x > 0")[1] == ">"
    assert read_equation("x == 1")[1] == "="


@pytest.mark.parametrize("text, words", [
    ("x^2", "use \\*\\* instead of \\^"),
    ("x = 3", "equation or a rule"),
    ("exp", "needs brackets"),
    ("foo(x)", "is not allowed"),
    ("x + ", "cannot read"),
])
def test_read_formula_errors(text, words):
    with pytest.raises(SimplydataError, match=words):
        read_formula(text)


def test_read_formula_did_you_mean_column():
    with pytest.raises(SimplydataError, match='Did you mean the column "time"'):
        read_formula("a * tme", columns=["time"])


def test_read_equation_errors():
    with pytest.raises(SimplydataError, match="no = sign"):
        read_equation("x + 1")
    with pytest.raises(SimplydataError, match="2 signs"):
        read_equation("0 <= x <= 4")


# ---------------------------------------------------------------------------- one name
def test_smallest_on_a_range(capsys):
    # R: optimize(function(x) x^2 - 4*x + 1, c(0, 5), tol = 1e-10) -> 2, -3
    r = row(find_smallest("x**2 - 4*x + 1", between=[0, 5]))
    out = capsys.readouterr().out
    assert "The smallest value of x**2 - 4*x + 1 between 0 and 5 is -3, at x = 2." in out
    assert "Brent's method on the range, as R optimize()" in out
    assert "several dips" not in out
    assert r["x"] == pytest.approx(2, abs=1e-8) and r["smallest value"] == pytest.approx(-3, abs=1e-12)
    assert r["settled"] == "yes" and r["method"] == "Brent's method on the range"


def test_several_dips_note(capsys):
    # R: optimize(function(x) x * sin(x), c(0, 15), tol = 1e-10) -> 4.91318042873019, -4.81446988971227
    r = row(find_smallest("x * sin(x)", between=[0, 15]))
    out = capsys.readouterr().out
    assert r["x"] == pytest.approx(4.91318042873019, abs=1e-7)
    assert r["smallest value"] == pytest.approx(-4.81446988971227, abs=1e-10)
    assert "several dips may have a lower one elsewhere" in out
    assert "x = 11.0804 gives a lower value" in out  # the scan; never a silent switch


def test_lowest_point_at_the_edge(capsys):
    r = row(find_smallest("x", between=[0, 5]))
    assert r["x"] == 0 and "at the end of the range" in capsys.readouterr().out


def test_biggest_with_known_values(capsys):
    # R: optimize(function(x) 2 * x * exp(-0.5 * x), c(0, 20), tol = 1e-10, maximum = TRUE)
    #    -> 2.00000000572146, 1.47151776468577
    r = row(find_biggest("a * x * exp(-b * x)", between=[0, 20], known={"a": 2, "b": 0.5}))
    out = capsys.readouterr().out
    assert "Using a = 2, b = 0.5 from known=." in out
    assert "The biggest value of a * x * exp(-b * x) between 0 and 20 is 1.47152, at x = 2." in out
    assert r["x"] == pytest.approx(2, abs=1e-7)
    assert r["biggest value"] == pytest.approx(1.47151776468577, abs=1e-12)


def test_biggest_with_a_fit_formula_table():
    marks = make_table(sdata.example("marks"))
    fit = marks.fit_formula("marks = a * hours + b")
    a, b = row(fit)["a"], row(fit)["b"]
    r = row(find_biggest("a * x + b - x**2", known=fit, between=[0, 50]))
    assert r["x"] == pytest.approx(a / 2, abs=1e-7)
    assert r["biggest value"] == pytest.approx(b + a ** 2 / 4, abs=1e-9)


def test_smallest_from_a_start(capsys):
    # R: optimize(function(x) (x - 3)^2 + 1, c(0.261803398874989, 7.43050983124842), tol = 1e-10) -> 3, 1
    # (the bracket printed by simplydata)
    r = row(find_smallest("(x - 3)**2 + 1", start=0))
    out = capsys.readouterr().out
    assert "Brent's method from a bracket around the start (x from 0.261803 to 7.4305)" in out
    assert r["x"] == pytest.approx(3, abs=1e-7) and r["smallest value"] == pytest.approx(1, abs=1e-12)


def test_no_smallest_from_a_start():
    with pytest.raises(SimplydataError, match="keeps getting smaller"):
        find_smallest("x**3", start=0)


# ---------------------------------------------------------------------------- several names
ROSENBROCK = "100*(y - x**2)**2 + (1 - x)**2"


def test_nelder_mead(capsys):
    # R: optim(c(-1.2, 1), rosen) -> 1.00026, 1.00051 (default reltol 1e-8 stops early);
    #    optim(that, rosen, method = "BFGS") -> 1.000002, 1.000005; the minimum is exactly (1, 1), value 0.
    # SciPy runs with tighter tolerances, so its answer is compared with the exact place to 1e-6.
    r = row(find_smallest(ROSENBROCK, start={"x": -1.2, "y": 1}))
    out = capsys.readouterr().out
    assert "Method: Nelder-Mead from y = 1, x = -1.2, as R optim() (its default method)." in out
    assert r["x"] == pytest.approx(1, abs=1e-6) and r["y"] == pytest.approx(1, abs=1e-6)
    assert r["x"] == pytest.approx(1.00000200746402, abs=1e-4)  # R optim then BFGS
    assert r["smallest value"] == pytest.approx(0, abs=1e-12)


def test_l_bfgs_b_with_limits(capsys):
    # R: optim(c(-1.2, 1), rosen, method = "L-BFGS-B", lower = c(-2, -1), upper = c(0.5, 2)) -> 0.5, 0.25; 0.25
    r = row(find_smallest(ROSENBROCK, start={"x": -1.2, "y": 1}, between={"x": [-2, 0.5], "y": [-1, 2]}))
    out = capsys.readouterr().out
    assert 'L-BFGS-B from y = 1, x = -1.2 with the limits in between=, as R optim(method = "L-BFGS-B")' in out
    assert r["x"] == pytest.approx(0.5, abs=1e-6) and r["y"] == pytest.approx(0.25, abs=1e-6)
    assert r["smallest value"] == pytest.approx(0.25, abs=1e-8)


@pytest.mark.parametrize("method", ["bfgs", "l-bfgs-b", "powell", "trust region", "slsqp"])
def test_expert_methods(method):
    # all should end near the exact minimum (1, 1); numerical slopes limit the gradient methods to ~1e-5
    r = row(find_smallest(ROSENBROCK, start={"x": -1.2, "y": 1}, method=method))
    assert r["x"] == pytest.approx(1, abs=1e-4) and r["y"] == pytest.approx(1, abs=1e-4)


def test_cobyla():
    # COBYLA is slow on Rosenbrock's valley, so a simple bowl: the exact answer is (1, -2), value 3
    r = row(find_smallest("(x - 1)**2 + 2*(y + 2)**2 + 3", start={"x": 0, "y": 0}, method="cobyla"))
    assert (r["x"], r["y"], r["smallest value"]) == pytest.approx((1, -2, 3), abs=1e-4)


def test_method_spelling_and_errors():
    r = row(find_smallest(ROSENBROCK, start={"x": -1.2, "y": 1}, method="Nelder Mead"))
    assert r["method"] == "Nelder-Mead"
    with pytest.raises(SimplydataError, match='Did you mean method="nelder-mead"'):
        find_smallest("x**2", between=[0, 5], method="nelder")


def test_several_names_need_start():
    with pytest.raises(SimplydataError, match="needs a start for each name"):
        find_smallest(ROSENBROCK)
    with pytest.raises(SimplydataError, match="between= needs a range for each"):
        find_smallest(ROSENBROCK, between=[0, 1])


def test_no_range_and_no_start():
    with pytest.raises(SimplydataError, match=r"needs between=\[low, high\] or start="):
        find_smallest("x**2")


# ---------------------------------------------------------------------------- rules
def test_non_linear_rule_slsqp(capsys):
    # R: nloptr(c(0, 0), ..., algorithm = "NLOPT_LD_SLSQP") with x^2 + y^2 <= 1
    #    -> 0.447213595500034, 0.894427190999972; 1.52786404500021 (exactly (1, 2) / sqrt(5))
    r = row(find_smallest("(x - 1)**2 + (y - 2)**2", rules=["x**2 + y**2 <= 1"], start={"x": 0, "y": 0}))
    out = capsys.readouterr().out
    assert "SLSQP (sequential quadratic programming) under the rules" in out
    assert "because the formula or a rule is not straight-line" in out
    assert "x**2 + y**2 <= 1 is reached exactly." in out
    assert r["x"] == pytest.approx(0.447213595500034, abs=1e-7)
    assert r["y"] == pytest.approx(0.894427190999972, abs=1e-7)
    assert r["smallest value"] == pytest.approx(1.52786404500021, abs=1e-9)


def test_linear_program_and_shadow_prices(capsys):
    # R: lp("max", c(3, 2), rbind(c(1, 1), c(1, 3)), c("<=", "<="), c(4, 6), compute.sens = TRUE)
    #    objval 12, solution 4 0, duals 3 0 0 -1 (two rules, then the reduced costs of x and y;
    #    lp() assumes x, y >= 0, which simplydata needs as rules, and y's reduced cost -1 is the
    #    value of one more on "y >= 0")
    r = row(find_biggest("3*x + 2*y", rules=["x + y <= 4", "x + 3*y <= 6", "x >= 0", "y >= 0"]))
    out = capsys.readouterr().out
    assert "The biggest value of 3*x + 2*y under the 4 rules is 12, at x = 4, y = 0." in out
    assert ("Method: linear programming (HiGHS), as R lpSolve::lp(), because the formula and every rule are "
            "straight-line (linear).") in out
    assert "x + y <= 4 and y >= 0 are reached exactly" in out
    assert (r["x"], r["y"], r["biggest value"]) == pytest.approx((4, 0, 12), abs=1e-9)
    assert r["value of one more: x + y <= 4"] == pytest.approx(3, abs=1e-9)
    assert r["value of one more: x + 3*y <= 6"] == pytest.approx(0, abs=1e-9)
    assert r["value of one more: y >= 0"] == pytest.approx(-1, abs=1e-9)


def test_linear_program_smallest_with_at_least_rules():
    # R: lp("min", c(2, 3), rbind(c(1, 1), c(1, 2)), c(">=", ">="), c(4, 6), compute.sens = TRUE)
    #    objval 10, solution 2 2, duals 1 1 0 0
    r = row(find_smallest("2*x + 3*y", rules=["x + y >= 4", "x + 2*y >= 6", "x >= 0", "y >= 0"]))
    assert (r["x"], r["y"], r["smallest value"]) == pytest.approx((2, 2, 10), abs=1e-9)
    assert r["value of one more: x + y >= 4"] == pytest.approx(1, abs=1e-9)
    assert r["value of one more: x + 2*y >= 6"] == pytest.approx(1, abs=1e-9)


def test_whole_numbers(capsys):
    # R: lp("max", c(5, 4), rbind(c(6, 4), c(1, 2)), c("<=", "<="), c(24, 6), int.vec = 1:2) -> 20 at 4 0
    #    (without int.vec: 21 at 3, 1.5)
    rules = ["6*x + 4*y <= 24", "x + 2*y <= 6", "x >= 0", "y >= 0"]
    r = row(find_biggest("5*x + 4*y", rules=rules, whole_numbers=True))
    assert "mixed-integer linear programming (HiGHS), as R lpSolve::lp(int.vec = ...)" in capsys.readouterr().out
    assert (r["x"], r["y"], r["biggest value"]) == pytest.approx((4, 0, 20), abs=1e-9)
    relaxed = row(find_biggest("5*x + 4*y", rules=rules))
    assert (relaxed["x"], relaxed["y"], relaxed["biggest value"]) == pytest.approx((3, 1.5, 21), abs=1e-9)


def test_knapsack():
    # R: lp("max", c(10, 13, 7, 8), rbind(c(3, 4, 2, 3)), "<=", 7, all.bin = TRUE) -> 23 at 1 1 0 0
    r = row(find_biggest("10*a + 13*b + 7*c + 8*d", rules=["3*a + 4*b + 2*c + 3*d <= 7"],
                         between={n: [0, 1] for n in "abcd"}, whole_numbers=True))
    assert [r[n] for n in "abcd"] == pytest.approx([1, 1, 0, 0]) and r["biggest value"] == pytest.approx(23)


def test_infeasible_and_unbounded():
    with pytest.raises(SimplydataError, match="No values meet every rule at once"):
        find_smallest("x", rules=["x >= 5", "x <= 3"])
    with pytest.raises(SimplydataError, match="grow without end.*such as x >= 0"):
        find_biggest("x + y", rules=["x - y <= 1", "x >= 0"])


def test_whole_numbers_need_a_linear_formula():
    with pytest.raises(SimplydataError, match="straight-line"):
        find_smallest("x**2 + y", whole_numbers=True, start={"x": 1, "y": 1})


# ---------------------------------------------------------------------------- whole range
def test_whole_range_rastrigin(capsys):
    # the known answer: Rastrigin's function is 0 at (0, 0), its lowest point among many dips
    rastrigin = "20 + x**2 - 10*cos(2*pi*x) + y**2 - 10*cos(2*pi*y)"
    r = row(find_smallest(rastrigin, between={"x": [-5.12, 5.12], "y": [-5.12, 5.12]}, search="whole range"))
    out = capsys.readouterr().out
    assert "differential evolution over the whole range" in out and "seed=1" in out
    assert r["x"] == pytest.approx(0, abs=1e-6) and r["y"] == pytest.approx(0, abs=1e-6)
    assert r["smallest value"] == pytest.approx(0, abs=1e-9)


HIMMELBLAU_MINIMA = [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]


@pytest.mark.parametrize("method", ["differential evolution", "dual annealing", "shgo", "basin hopping", "direct"])
def test_whole_range_methods_himmelblau(method):
    # the known answer: Himmelblau's function has four lowest points, all with value 0
    r = row(find_smallest("(x**2 + y - 11)**2 + (x + y**2 - 7)**2", between={"x": [-5, 5], "y": [-5, 5]},
                          search="whole range", method=method, start={"x": 0, "y": 0}))
    assert r["smallest value"] == pytest.approx(0, abs=1e-8)
    assert min(math.dist((r["x"], r["y"]), m) for m in HIMMELBLAU_MINIMA) < 1e-4


def test_whole_range_needs_two_ends():
    with pytest.raises(SimplydataError, match="needs a range with two ends"):
        find_smallest("x**2 + y**2", between={"x": [-1, 1], "y": [0, None]}, search="whole range")


# ---------------------------------------------------------------------------- solve
def test_solve_on_a_range(capsys):
    # R: uniroot(function(x) x^3 - 2*x - 5, c(0, 3), tol = 1e-12)$root -> 2.09455148154233
    r = row(solve("x**3 - 2*x = 5", between=[0, 3]))
    out = capsys.readouterr().out
    assert "x**3 - 2*x = 5 between 0 and 3 when x = 2.09455." in out
    assert "Method: Brent's method, as R uniroot(); every sign change on 1000 steps" in out
    assert r["x"] == pytest.approx(2.09455148154233, abs=1e-12)
    assert abs(r["left minus right"]) < 1e-12


def test_solve_finds_every_crossing():
    # R: rootSolve::uniroot.all(function(x) sin(x) - 0.5, c(0, 10), n = 1000, tol = 1e-12)
    expected = [0.523598775598299, 2.617993877991494, 6.806784082777885, 8.901179185171081]
    found = [r["x"] for r in rows(solve("sin(x) = 0.5", between=[0, 10]))]
    assert found == pytest.approx(expected, abs=1e-11)


def test_solve_left_out_jumps(capsys):
    found = rows(solve("1 / x = 0.5", between=[-3, 3.1]))  # 0 is not one of the steps
    assert [r["x"] for r in found] == pytest.approx([2], abs=1e-11)
    assert "was a jump" in capsys.readouterr().out


def test_solve_from_a_start(capsys):
    r = row(solve("x**3 - 2*x = 5", start=1))
    assert r["x"] == pytest.approx(2.09455148154233, abs=1e-12)
    assert "the secant method" in capsys.readouterr().out


def test_solve_polynomial_with_complex_solutions(capsys):
    # R: polyroot(c(-5, -2, 0, 1)) -> 2.09455148154233+0i, -1.04727574077116+-1.13593988908893i
    found = rows(solve("x**3 - 2*x = 5"))
    out = capsys.readouterr().out
    assert "2 more solutions are not real numbers: -1.04728 + 1.13594i and -1.04728 - 1.13594i." in out
    assert "as R polyroot()" in out
    assert found[0]["x"] == pytest.approx(2.09455148154233, abs=1e-12) and found[0]["real number"] == "yes"
    assert [r["solution"] for r in found[1:]] == ["-1.04728 + 1.13594i", "-1.04728 - 1.13594i"]
    assert all(math.isnan(r["x"]) for r in found[1:])


def test_solve_polynomial_all_real():
    # R: polyroot(c(6, -5, 1)) -> 2, 3
    found = rows(solve("x**2 + 6 = 5*x"))
    assert [r["x"] for r in found] == pytest.approx([2, 3], abs=1e-12)
    assert "solution" not in found[0]


def test_solve_system(capsys):
    # R: nleqslv::nleqslv(c(0, 1), function(p) c(p[1] + p[2] - 3, p[1] * p[2] - 2))$x -> 2 1
    r = row(solve(["x + y = 3", "x * y = 2"], start={"x": 0, "y": 1}))
    assert "Powell's hybrid method (MINPACK hybrd) from the start, as R nleqslv::nleqslv()" in capsys.readouterr().out
    assert (r["x"], r["y"]) == pytest.approx((2, 1), abs=1e-10)
    assert abs(r["left minus right: x * y = 2"]) < 1e-10


def test_solve_linear_system(capsys):
    # R: solve(rbind(c(1, 1, 1), c(0, 2, 5), c(2, 5, -1)), c(6, -4, 27)) -> 5 3 -2
    r = row(solve(["x + y + z = 6", "2*y + 5*z = -4", "2*x + 5*y - z = 27"]))
    out = capsys.readouterr().out
    assert "solved exactly" in out and "as R solve()" in out
    assert (r["x"], r["y"], r["z"]) == pytest.approx((5, 3, -2), abs=1e-12)
    with pytest.raises(SimplydataError, match="do not pin down one answer"):
        solve(["x + y = 3", "2*x + 2*y = 6"])


def test_solve_errors():
    with pytest.raises(SimplydataError, match="needs an = sign"):
        solve("x**2 + 1")
    with pytest.raises(SimplydataError, match="never cross between 0 and 3; try a wider between="):
        solve("x**2 = -1", between=[0, 3])
    with pytest.raises(SimplydataError, match="more names to find"):
        solve("x + y = 3")
    with pytest.raises(SimplydataError, match="works with = only"):
        solve("x <= 3")
    with pytest.raises(SimplydataError, match="needs between="):
        solve("exp(x) = 2")


def test_solve_with_known():
    known = make_table(pd.DataFrame({"a": [2.0], "b": [0.5]}))
    r = row(solve("a * exp(-b * x) = 1", between=[0, 10], known=known))
    assert r["x"] == pytest.approx(2 * math.log(2), abs=1e-12)


# ---------------------------------------------------------------------------- engines and options
def test_fast_gives_a_polars_table():
    pl = pytest.importorskip("polars")
    result = find_smallest("x**2 - 2*x", between=[0, 5], fast=True)
    assert isinstance(result._frame.to_native(), pl.DataFrame)
    assert result._frame["x"].to_list()[0] == pytest.approx(1, abs=1e-8)
    assert isinstance(solve("x**2 = 4", fast=True)._frame.to_native(), pl.DataFrame)


def test_unknown_option_is_friendly():
    with pytest.raises(SimplydataError, match="Did you mean between="):
        find_smallest("x**2", betwen=[0, 1])


def test_equality_rule_and_strict_note(capsys):
    # the closest point to (0, 0) on the line x + y = 2 is (1, 1); exact answer, value 2
    r = row(find_smallest("x**2 + y**2", rules=["x + y = 2", "x > -5"], start={"x": 0, "y": 0}))
    out = capsys.readouterr().out
    assert "< and > in rules are treated as <= and >=" in out
    assert (r["x"], r["y"], r["smallest value"]) == pytest.approx((1, 1, 2), abs=1e-7)


def test_whole_range_with_rules_and_whole_numbers():
    # inside the circle x^2 + y^2 <= 4 the lowest point of x + y is -2*sqrt(2), at (-sqrt(2), -sqrt(2))
    r = row(find_smallest("x + y + 0*x**2", rules=["x**2 + y**2 <= 4"], between={"x": [-3, 3], "y": [-3, 3]},
                          search="whole range"))
    assert r["smallest value"] == pytest.approx(-2 * math.sqrt(2), abs=1e-5)
    # whole numbers anywhere in the range: (x - 2.4)^2 + (y + 1.6)^2 is lowest at x = 2, y = -2
    r = row(find_smallest("(x - 2.4)**2 + (y + 1.6)**2", between={"x": [-5, 5], "y": [-5, 5]},
                          search="whole range", whole_numbers=True))
    assert (r["x"], r["y"]) == (2, -2)
    assert r["smallest value"] == pytest.approx(0.32, abs=1e-12)
