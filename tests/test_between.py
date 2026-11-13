"""Tests for values between known points: table.values_at, table.map_of and
fill_empty_cells(with_the="curve between neighbours").

Every printed number is checked against R 4.6.2 (values written in below, with the R call
that made them; R packages pracma 2.4, akima, interp, zoo from CRAN), on pandas and Polars
tables. The R script is kept with the builder's notes; each value says its call.
"""

import math

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table.between import (SmoothingSpline, _curve_through, curve_between_neighbours, join_many_directions,
                                      join_one_direction, smoothing_spline)
from simplydata.table.table import Table

ENGINES = ["pandas", "polars"]
TOL = 1e-8  # interpolation: R and SciPy follow the same formulas


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return Table(pl.from_pandas(data))
    return Table(data)


def frame(result):
    return result._frame.to_pandas() if hasattr(result._frame, "to_pandas") else result._frame


# Applia in sdata.example("growth"): R  x <- seq(2000, 2024, by = 2); y <- c(12.0, ..., 24.1)
YEARS = np.arange(2000, 2025, 2.0)
APPLIA = np.array([12.0, 12.6, 13.4, 14.4, 15.1, 16.2, 17.0, 18.1, 19.6, 20.4, 21.9, 23.2, 24.1])
AT = [2001, 2003, 2011, 2023.5, 2004, 2005]
R_ONE_DIRECTION = {
    # approx(x, y, at)
    "straight lines": [12.3, 13, 16.6, 23.875, 13.4, 13.9],
    # approx(x, y, at, method = "constant", f = 0)
    "steps": [12, 12.6, 16.2, 23.2, 13.4, 13.4],
    # sapply(at, function(a) y[which.min(abs(x - a))])  (2005 is halfway: the lower point, 2004)
    "nearest": [12, 12.6, 16.2, 24.1, 13.4, 13.4],
    # splinefun(x, y, method = "natural")(at)
    "smooth curve": [12.2883992804, 12.9598021587, 16.6240108645, 23.892659406, 13.4, 13.9223920849],
    # pracma::pchip(x, y, at)
    "curve without overshoot": [12.2767857143, 12.9746031746, 16.6, 23.9107954545, 13.4, 13.9081699346],
    # akima::aspline(x, y, xout = at)$y
    "akima": [12.275, 12.9775, 16.6, 23.915625, 13.4, 13.8975],
}


# ====================================================================== the engine, against R
@pytest.mark.parametrize("how", list(R_ONE_DIRECTION))
def test_one_direction_matches_r(how):
    got = join_one_direction(YEARS, APPLIA, AT, how)
    assert np.allclose(got, R_ONE_DIRECTION[how], rtol=0, atol=1e-9)


def test_ends_as_r():
    # approx(x, y, c(1990, 2030), rule = 2)$y
    assert np.allclose(join_one_direction(YEARS, APPLIA, [1990, 2030], outside="nearest end"), [12, 24.1])
    assert np.isnan(join_one_direction(YEARS, APPLIA, [1990, 2030])).all()
    # approx(x, y, 2024, method = "constant", f = 0)$y: the last point keeps its own value
    assert join_one_direction(YEARS, APPLIA, [2024], "steps")[0] == 24.1
    # splinefun(x, y, method = "natural")(c(1995, 2030)): straight on beyond the ends
    got = join_one_direction(YEARS, APPLIA, [1995, 2030], "smooth curve", outside="extend")
    assert np.allclose(got, [10.5773381303, 26.5739596026], atol=1e-9)


def test_periodic_matches_r():
    # m <- 1:13; t <- Sunnyvale temperatures with January repeated at 13
    # splinefun(m, t, method = "periodic")(c(1.5, 6.25, 12.5)) and (c(14.5, 0.5))
    m = np.arange(1, 14.0)
    t = np.array([8.3, 10.3, 11.0, 16.5, 20.0, 24.0, 26.5, 22.8, 21.8, 17.5, 13.9, 10.9, 8.3])
    assert np.allclose(join_one_direction(m, t, [1.5, 6.25, 12.5], "periodic"),
                       [9.16033653846, 25.0723137019, 9.14764423077], atol=1e-9)
    assert np.allclose(join_one_direction(m, t, [14.5, 0.5], "periodic", outside="extend"),
                       [10.4235096154, 9.14764423077], atol=1e-9)


def test_not_a_knot_and_makima_self_checks():
    # No R function: check the defining properties by hand.
    curve = _curve_through(YEARS, APPLIA, "not-a-knot")
    third = curve.derivative(3)
    # not-a-knot: the third derivative does not jump at the second and the second-to-last point
    assert third(YEARS[1] - 1e-9) == pytest.approx(third(YEARS[1] + 1e-9), rel=1e-6)
    assert third(YEARS[-2] - 1e-9) == pytest.approx(third(YEARS[-2] + 1e-9), rel=1e-6)
    assert np.allclose(curve(YEARS), APPLIA)
    # makima (Akima 2019 / MATLAB): slope at point i from the slopes d of the 4 pieces around it
    curve = _curve_through(YEARS, APPLIA, "makima")
    d = np.diff(APPLIA) / np.diff(YEARS)
    i = 5
    w1 = abs(d[i + 1] - d[i]) + abs(d[i + 1] + d[i]) / 2
    w2 = abs(d[i - 1] - d[i - 2]) + abs(d[i - 1] + d[i - 2]) / 2
    assert curve.derivative()(YEARS[i]) == pytest.approx((w1 * d[i - 1] + w2 * d[i]) / (w1 + w2), rel=1e-12)


def test_curve_without_overshoot_keeps_rises_rising():
    x = np.array([0, 1, 2, 3, 4.0])
    y = np.array([0, 0, 1, 1, 1.0])
    xs = np.linspace(0, 4, 401)
    assert np.all(np.diff(join_one_direction(x, y, xs, "curve without overshoot")) >= -1e-12)
    assert join_one_direction(x, y, xs, "smooth curve").min() < -1e-3  # the natural spline overshoots


# ---------------------------------------------------------------------- smoothing spline
def test_smoothing_spline_fixed_lambda_matches_r():
    # f <- smooth.spline(x, y, all.knots = TRUE, lambda = 1e-4); predict(f, at)$y
    # R solves the same penalised problem in a B-spline basis; the answers agree to about 1e-5.
    got = join_one_direction(YEARS, APPLIA, AT, smoothness=1e-4)
    r = [12.2934434513, 13.0157650071, 16.5740925874, 23.9026633748, 13.4313361049, 13.8735791048]
    assert np.allclose(got, r, atol=2e-5)
    # predict(f, c(1990, 2030))$y: straight on beyond the ends
    got = join_one_direction(YEARS, APPLIA, [1990, 2030], smoothness=1e-4, outside="extend")
    assert np.allclose(got, [8.78556599928, 27.115561479], atol=5e-4)


GCV_X = np.array([0.133903, 0.617863, 1.079436, 1.255551, 1.765568, 1.862176, 2.016819, 2.059746, 2.121425,
                  2.655087, 2.672207, 3.403490, 3.721239, 3.800352, 3.823880, 3.841037, 3.861141, 4.112744,
                  4.820801, 4.935413, 4.976992, 5.728534, 5.995658, 6.291140, 6.516738, 6.607978, 6.684667,
                  6.870228, 7.176185, 7.237109, 7.698414, 7.774452, 7.942399, 8.273733, 8.696908, 8.983897,
                  9.082078, 9.347052, 9.446753, 9.919061])
GCV_Y = np.array([0.409197, 0.813935, 0.904061, 0.353915, 1.167040, 0.941010, 0.855431, 0.441601, 0.708751,
                  0.592923, 0.859943, -0.289750, -0.431427, -0.628278, -1.043687, -0.768291, -0.777332,
                  -0.843330, -0.664122, -0.746280, -1.014553, -0.602656, -0.074493, 0.174954, 0.024808,
                  0.106864, 0.500158, 0.784462, 0.745252, 1.080024, 1.107356, 0.813231, 1.098430, 0.574381,
                  1.095188, 1.020856, 0.225865, -0.235593, 0.148943, -0.514917])


def test_smoothing_spline_gcv_matches_r():
    # set.seed(1); xx <- round(sort(runif(40, 0, 10)), 6); yy <- round(sin(xx) + rnorm(40, sd = 0.3), 6)
    # g <- smooth.spline(xx, yy, all.knots = TRUE): lambda 0.000197841078495, df 8.410149
    # The GCV curve is flat near its lowest point, so the two searches stop a little apart:
    # values within 1e-3, lambda within 1 %.
    spline = smoothing_spline(GCV_X, GCV_Y)
    assert spline.lam == pytest.approx(0.000197841078495, rel=0.01)
    assert spline.df == pytest.approx(8.410149217828652, abs=0.01)
    r = [0.63815878794, 0.794771395298, -0.574718025973, 0.521633177563]  # predict(g, c(0.5, 2, 5.5, 9))$y
    assert np.allclose(spline([0.5, 2, 5.5, 9]), r, atol=1e-3)
    # growth, Applia: smooth.spline(x, y, all.knots = TRUE): lambda 0.0029062685462, df 4.0308516
    spline = smoothing_spline(YEARS, APPLIA)
    assert spline.df == pytest.approx(4.0308516162634, abs=0.01)
    assert np.allclose(spline([2001, 2003, 2011]), [12.2633508089467, 13.0542882126724, 16.6317113148074],
                       atol=1e-3)


def test_smoothing_spline_with_ties_matches_r():
    # m <- read.csv(".../marks.csv"); 54 rows, 40 different hours (ties are averaged, and R's GCV
    # counts their spread): smooth.spline(m$hours, m$marks, all.knots = TRUE)
    # lambda 0.0808156517, df 2.7685814228; predict(f, c(2, 5, 8.5))$y
    marks = sdata.example("marks").dropna(subset=["hours", "marks"])
    spline = smoothing_spline(marks["hours"], marks["marks"])
    assert spline.df == pytest.approx(2.7685814228, abs=0.005)
    assert np.allclose(spline([2, 5, 8.5]), [46.8065240231, 64.3289340207, 80.4701015264], atol=1e-3)
    # smooth.spline(..., lambda = 0.01): predict(f, c(2, 5, 8.5))$y
    fixed = smoothing_spline(marks["hours"], marks["marks"], 0.01)
    assert np.allclose(fixed([2, 5, 8.5]), [46.3672143365, 65.6487989785, 79.9461598679], atol=1e-4)
    assert isinstance(fixed, SmoothingSpline)


# ---------------------------------------------------------------------- two directions
SX = np.array([1.8488, 7.0237, 5.7333, 1.6805, 9.4384, 9.4347, 1.2916, 8.3345, 4.6802, 5.4998, 5.5267, 2.3889,
               7.6051, 1.8082, 4.0528])
SY = np.array([8.5355, 9.764, 2.2583, 4.4481, 0.7498, 6.619, 3.8755, 8.3689, 1.505, 3.4727, 4.8877, 1.4925,
               3.5706, 9.6264, 1.3237])
SZ = np.array([2.5396, 7.5326, 0.7722, 1.7415, 0.6941, 6.2349, 1.4618, 7.8618, -0.2951, 1.2042, 2.0149, 1.0401,
               3.6847, 2.7126, -0.2538])
WANTED = np.column_stack([[3, 5, 6.5, 2], [4, 5, 2.5, 7]])


def test_scattered_triangles_match_r():
    # set.seed(2); sx <- round(runif(15, 0, 10), 4); sy <- ...; sz <- round(sx * sy / 10 + sin(sx), 4)
    # interp::interp(sx, sy, sz, xo = c(3, 5, 6.5, 2), yo = c(4, 5, 2.5, 7), linear = TRUE, output = "points")$z
    got, kind = join_many_directions(np.column_stack([SX, SY]), SZ, WANTED)
    assert kind == "scattered"
    assert np.allclose(got, [1.48256521534, 2.01079440685, 1.54420225495, 2.25028652792], atol=1e-9)


def test_thin_plate_spline_matches_r_by_hand():
    # R: K = r^2 log r between the points, P = cbind(1, sx, sy);
    # solve(rbind(cbind(K + lambda * diag(n), P), cbind(t(P), 0)), c(sz, 0, 0, 0)), then evaluated
    points = np.column_stack([SX, SY])
    got, _ = join_many_directions(points, SZ, WANTED, "smooth surface")
    assert np.allclose(got, [1.36732411106, 1.80350032414, 1.60357772281, 2.27476430755], atol=1e-8)
    got, _ = join_many_directions(points, SZ, WANTED, "smooth surface", smoothness=0.5)
    assert np.allclose(got, [1.35456397673, 1.85755575086, 1.54163586196, 2.22263635005], atol=1e-8)


# ====================================================================== values_at
def test_values_at_one_direction_for_each(engine, capsys):
    table = make(sdata.example("growth"), engine)
    result = table.values_at("population", at={"year": [2001, 2003, 2011]}, for_each="country")
    out = capsys.readouterr().out
    assert ('Estimated "population" at 3 values of "year" for each of the 4 groups in "country", joined by '
            "straight lines between the known points (linear interpolation, as R approx()).") in out
    got = frame(result)
    assert list(got.columns) == ["year", "country", "population"]
    assert len(got) == 12
    applia = got[got["country"] == "Applia"]
    assert np.allclose(applia["population"], [12.3, 13.0, 16.6])
    assert applia["year"].tolist() == [2001, 2003, 2011]
    assert len(table) == 52  # the table itself is not changed


@pytest.mark.parametrize("how", ["smooth curve", "curve without overshoot", "steps", "nearest", "akima"])
def test_values_at_each_way(engine, how, capsys):
    table = make(sdata.example("growth"), engine).keep_rows("country", equal_to="Applia")
    result = frame(table.values_at("population", at={"year": AT}, joined_by=how))
    assert np.allclose(result["population"], R_ONE_DIRECTION[how], atol=1e-9)
    assert how.split()[0] in capsys.readouterr().out


def test_values_at_other_names_and_spellings():
    table = Table(sdata.example("growth")).keep_rows("country", equal_to="Applia")
    assert np.allclose(frame(table.values_at("population", at={"year": AT}, joined_by="spline"))["population"],
                       R_ONE_DIRECTION["smooth curve"])
    assert np.allclose(frame(table.values_at("population", at={"year": AT}, joined_by="PCHIP"))["population"],
                       R_ONE_DIRECTION["curve without overshoot"])
    assert np.allclose(frame(table.values_at("population", at={"year": 2001}))["population"], [12.3])


def test_values_at_outside(capsys):
    table = Table(sdata.example("growth")).keep_rows("country", equal_to="Applia")
    capsys.readouterr()
    got = frame(table.values_at("population", at={"year": [1990, 2001, 2030]}))["population"]
    assert np.isnan(got[0]) and np.isnan(got[2]) and got[1] == pytest.approx(12.3)
    out = capsys.readouterr().out
    assert ('2 places were outside 2000 to 2024 in "year", so they are empty; outside="nearest end" fills them'
            in out)
    got = frame(table.values_at("population", at={"year": [1990, 2030]}, outside="nearest end"))["population"]
    assert got.tolist() == [12.0, 24.1]
    got = frame(table.values_at("population", at={"year": [1995, 2030]}, joined_by="smooth curve",
                                outside="extend"))["population"]
    assert np.allclose(got, [10.5773381303, 26.5739596026])
    assert "guesswork" in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="nearest end"):
        table.values_at("population", at={"year": [2030]}, outside="extend")


def test_values_at_smoothing(engine, capsys):
    table = make(sdata.example("growth"), engine).keep_rows("country", equal_to="Applia")
    capsys.readouterr()
    got = frame(table.values_at("population", at={"year": [2001, 2003, 2011]}, smoothness="auto"))
    assert np.allclose(got["population"], [12.2633508089467, 13.0542882126724, 16.6317113148074], atol=1e-3)
    out = capsys.readouterr().out
    assert "smoothing spline" in out and "generalised cross-validation" in out
    assert "4.03 equivalent degrees of freedom" in out and "smooth.spline" in out
    got = frame(table.values_at("population", at={"year": AT}, smoothness=1e-4))
    assert np.allclose(got["population"], [12.2934434513, 13.0157650071, 16.5740925874, 23.9026633748,
                                           13.4313361049, 13.8735791048], atol=2e-5)
    assert "lambda = 0.0001" in capsys.readouterr().out


def test_values_at_ties_and_empty_cells(capsys):
    data = pd.DataFrame({"x": [1, 2, 2, 3, None, 4], "y": [1.0, 2.0, 4.0, 5.0, 9.0, None]})
    table = Table(data)
    got = frame(table.values_at("y", at={"x": [1.5, 2]}))["y"]
    # approx(c(1, 2, 2, 3), c(1, 2, 4, 5), c(1.5, 2))$y: ties are averaged (ties = mean)
    assert np.allclose(got, [2.0, 3.0])
    out = capsys.readouterr().out
    assert "2 rows with an empty cell" in out
    assert "1 place had several known values; their average was used" in out


def test_values_at_two_directions(engine, capsys):
    table = make(sdata.example("island"), engine)
    got = frame(table.values_at("height", at={"east": [1.5, 2.5, 7.2], "north": [3, 4.5, 11.9]}))
    # by hand in R on the 15 x 15 grid: H[e0, n0] (1 - de)(1 - dn) + ... (bilinear)
    assert np.allclose(got["height"], [0.5, 14.25, 37.72], atol=1e-9)
    assert list(got.columns) == ["east", "north", "height"]
    assert "full grid (bilinear interpolation)" in capsys.readouterr().out
    got = frame(table.values_at("height", at={"east": [1.5], "north": [3]}, joined_by="smooth surface"))
    assert "thin-plate spline" in capsys.readouterr().out
    assert np.isfinite(got["height"][0])
    with pytest.raises(SimplydataError, match="same length"):
        table.values_at("height", at={"east": [1, 2], "north": [1, 2, 3]})


def test_values_at_scattered_table(engine, capsys):
    data = pd.DataFrame({"east": SX, "north": SY, "depth": SZ})
    table = make(data, engine)
    got = frame(table.values_at("depth", at={"east": [3, 5, 6.5, 2, 20], "north": [4, 5, 2.5, 7, 20]}))
    assert np.allclose(got["depth"][:4], [1.48256521534, 2.01079440685, 1.54420225495, 2.25028652792], atol=1e-9)
    assert np.isnan(got["depth"][4])
    out = capsys.readouterr().out
    assert "flat triangles" in out and "interp::interp(linear = TRUE)" in out
    assert "outside the outline of the known points" in out


def test_values_at_dates():
    data = pd.DataFrame({"day": pd.to_datetime(["2026-01-01", "2026-01-11", "2026-01-21"]), "level": [1.0, 2.0, 4.0]})
    got = frame(Table(data).values_at("level", at={"day": ["2026-01-06", "2026-01-16"]}))
    assert np.allclose(got["level"], [1.5, 3.0])
    assert str(got["day"][0].date()) == "2026-01-06"


def test_values_at_friendly_errors():
    table = Table(sdata.example("growth"))
    with pytest.raises(SimplydataError, match="at="):
        table.values_at("population")
    with pytest.raises(SimplydataError, match='Did you mean joined_by="steps"'):
        table.values_at("population", at={"year": [2001]}, joined_by="stepz")
    with pytest.raises(SimplydataError, match="Did you mean"):
        table.values_at("population", at={"yaer": [2001]})
    with pytest.raises(SimplydataError, match="numbers"):
        table.values_at("country", at={"year": [2001]})
    with pytest.raises(SimplydataError, match="smoothness="):
        table.values_at("population", at={"year": [2001]}, smoothness=-1)
    with pytest.raises(SimplydataError, match="leave out"):
        table.values_at("population", at={"year": [2001]}, joined_by="steps", smoothness="auto")
    with pytest.raises(SimplydataError, match="first and last"):
        table.keep_rows("country", equal_to="Applia").values_at("population", at={"year": [2001]}, joined_by="periodic")
    with pytest.raises(SimplydataError, match="two or more directions"):
        table.values_at("population", at={"year": [2001]}, joined_by="smooth surface")


# ====================================================================== map_of
def test_map_of(engine, capsys):
    table = make(sdata.example("island"), engine)
    got = frame(table.map_of("height", across=["east", "north"], steps=5))
    assert len(got) == 25 and list(got.columns) == ["east", "north", "height"]
    assert got["east"].tolist()[:5] == [0.0] * 5
    assert np.allclose(got["north"][:5], [0, 3.5, 7, 10.5, 14])
    # grid points on the known grid keep their heights; (0, 3.5) is halfway between -3 and -1
    assert got["height"][0] == -6 and got["height"][1] == pytest.approx(-2.0)
    assert 'at 5 x 5 evenly spaced places across "east" and "north"' in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="steps="):
        table.map_of("height", across=["east", "north"], steps=1)


def test_map_of_scattered_outline(capsys):
    table = Table(pd.DataFrame({"east": SX, "north": SY, "depth": SZ}))
    got = frame(table.map_of("depth", across=["east", "north"], steps=[4, 3]))
    assert len(got) == 12
    assert got["depth"].isna().sum() > 0
    assert "outside the outline of the known points" in capsys.readouterr().out


# ====================================================================== fill_empty_cells
def test_curve_between_neighbours_matches_zoo():
    # library(zoo); v <- c(1, 4, NA, 16, NA, NA, 49, 64, NA); xi <- c(1, 2, 3, 4, 5.5, 6, 7, 8, 9)
    # na.spline(v, x = xi, method = "natural")  (zoo also extends to the end; simplydata leaves it empty)
    v = np.array([1, 4, np.nan, 16, np.nan, np.nan, 49, 64, np.nan])
    got = curve_between_neighbours(v, np.array([1, 2, 3, 4, 5.5, 6, 7, 8, 9]))
    assert np.allclose(got[:8], [1, 4, 8.94543147208, 16, 30.1700507614, 35.9018612521, 49, 64], atol=1e-9)
    assert np.isnan(got[8])
    # na.spline(c(NA, 2, 5, NA, 4, NA, NA, 10, 11), method = "natural")
    got = curve_between_neighbours(np.array([np.nan, 2, 5, np.nan, 4, np.nan, np.nan, 10, 11]))
    assert np.isnan(got[0])
    assert np.allclose(got[1:], [2, 5, 4.8826142132, 4, 5.18443316413, 7.71912013536, 10, 11], atol=1e-9)


def test_fill_empty_cells_with_curves(engine, capsys):
    data = pd.DataFrame({"shop": ["a"] * 9 + ["b"] * 4,
                         "day": [1, 2, 3, 4, 5.5, 6, 7, 8, 9, 1, 2, 3, 4],
                         "sales": [1, 4, None, 16, None, None, 49, 64, None, 1, None, 3, 4]})
    table = make(data, engine)
    table.fill_empty_cells("sales", with_the="curve between neighbours", sorted_by="day", for_each="shop")
    got = frame(table)["sales"]
    assert np.allclose(got[:8], [1, 4, 8.94543147208, 16, 30.1700507614, 35.9018612521, 49, 64], atol=1e-9)
    assert np.isnan(got[8])
    out = capsys.readouterr().out
    assert "Filled 4 cells in sales on a smooth curve" in out and "zoo::na.spline" in out
    assert "1 cell stayed empty" in out
    table.undo()
    table.fill_empty_cells("sales", with_the="curve without overshoot", sorted_by="day", for_each="shop")
    assert "never overshoots" in capsys.readouterr().out
    # shop b: pracma::pchip(c(1, 3, 4), c(1, 3, 4), 2) = 2
    assert frame(table)["sales"][10] == pytest.approx(2.0)
