"""Values between known points: join the dots, then read values off the curve or surface.

table.values_at("population", at={"year": [2001, 2003]}) estimates a column at places where
it was not measured, by joining the known points. The way they are joined is said in words
(joined_by=) and printed with its maths name and its R function, so it can be checked there.

One direction (one column in at=):

* "straight lines" (default): straight lines between neighbours, linear interpolation,
  NumPy interp, as R approx(). Several values at the same place are averaged first, as
  approx(ties = mean).
* "smooth curve": a cubic spline through every point with natural ends (no bend at the
  first and last point), SciPy CubicSpline(bc_type="natural"), as R splinefun(method =
  "natural"). Extending it beyond the points carries on the end slope in a straight line,
  as R.
* "curve without overshoot": the piecewise cubic Hermite curve of Fritsch and Carlson
  (PCHIP), SciPy PchipInterpolator, as MATLAB pchip() and R pracma::pchip(). It never goes
  above or below its neighbours, so rises keep rising. (R's splinefun(method = "monoH.FC")
  is a different monotone spline and gives other values.)
* "steps": the value stays until the next known point, as R approx(method = "constant",
  f = 0); at the last point its own value.
* "nearest": the value of the nearest known point; a place exactly halfway takes the
  lower one (as R y[which.min(abs(x - x0))] on sorted points, and SciPy interp1d(kind=
  "nearest")).
* experts: "akima" (Akima 1970, SciPy Akima1DInterpolator, as R akima::aspline()),
  "makima" (modified Akima, as MATLAB makima()), "not-a-knot" (cubic spline with
  not-a-knot ends, as MATLAB spline()), "periodic" (cubic spline for a cycle, the first
  and last values equal, as R splinefun(method = "periodic")).
* smoothness="auto" or a number: a smoothing spline that does NOT pass through every
  point. With x rescaled to run from 0 to 1 (as R does), it makes
  sum (y - curve)**2 + smoothness * integral(curve''**2)  as small as possible, with a knot
  at every distinct x (Reinsch's algorithm, written out here; R smooth.spline(x, y,
  all.knots = TRUE)); so smoothness=0.001 is R's lambda = 0.001. "auto" chooses it by
  generalised cross-validation over all the values, as R's default (SciPy's
  make_smoothing_spline is not used: its search for lambda stops at the number of points
  and it leaves out the spread of tied values, so it lands elsewhere). Several values at
  one x are averaged and weighted by how many there were, as R.

Two or more directions (at= with two or more columns):

* "straight lines": if the points form a full grid (every value of one column with every
  value of the others, like the island table), straight lines along each direction
  (bilinear; SciPy RegularGridInterpolator). Otherwise flat triangles between the points
  (Delaunay triangulation; SciPy griddata(method="linear"), as R interp::interp(linear =
  TRUE)). The sentence says which.
* "smooth surface": a thin-plate spline through the points (radial basis r**2 log r plus a
  flat plane; SciPy RBFInterpolator). smoothness= a number lets it pass near rather than
  through the points (added to the diagonal of the system).
* "nearest": the value of the nearest point.
* experts on full grids: "cubic", "quintic", "pchip" (SciPy RegularGridInterpolator).

outside= says what happens at places beyond the known points: "empty" (default, as R
approx(rule = 1)), "nearest end" (the value at the nearest end, as R approx(rule = 2); with
scattered points the value of the nearest point) or "extend" (the curve or surface carried
on; only for curves and surfaces, since extending is guesswork, a note says so).
"""

from __future__ import annotations

import math

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, choice_error, friendly
from .stat_tests import _list_words, _num, _skipped_note

# ------------------------------------------------------------------ vocabulary
JOINED_BY = ("straight lines", "smooth curve", "curve without overshoot", "steps", "nearest",
             "akima", "makima", "not-a-knot", "periodic")
JOINED_BY_MANY = ("straight lines", "smooth surface", "nearest", "cubic", "quintic", "pchip")
OUTSIDE = ("empty", "nearest end", "extend")
# other names people bring (R, SciPy, MATLAB, everyday words) -> the simplydata words
JOIN_ALIASES = {
    "straight line": "straight lines", "lines": "straight lines", "line": "straight lines",
    "linear": "straight lines", "bilinear": "straight lines", "approx": "straight lines",
    "smooth": "smooth curve", "curve": "smooth curve", "spline": "smooth curve", "natural": "smooth curve",
    "natural spline": "smooth curve", "cubic spline": "smooth curve",
    "curve without overshooting": "curve without overshoot", "monotone": "curve without overshoot",
    "no overshoot": "curve without overshoot",
    "step": "steps", "constant": "steps", "previous": "steps",
    "nearest point": "nearest", "nearest neighbour": "nearest", "nearest neighbor": "nearest",
    "not a knot": "not-a-knot", "notaknot": "not-a-knot",
    "thin plate spline": "smooth surface", "thin-plate spline": "smooth surface", "surface": "smooth surface",
}
OUTSIDE_ALIASES = {"nearest": "nearest end", "nearest ends": "nearest end", "clamp": "nearest end",
                   "extrapolate": "extend", "none": "empty", "nan": "empty", "na": "empty"}
CURVES = ("smooth curve", "curve without overshoot", "akima", "makima", "not-a-knot", "periodic")

# what each way of joining is, in words, for the printed sentence
SAID = {
    "straight lines": "straight lines between the known points (linear interpolation, as R approx())",
    "smooth curve": "a smooth curve through every known point (a natural cubic spline, as R "
                    'splinefun(method = "natural"))',
    "curve without overshoot": "a smooth curve that never overshoots its neighbours (PCHIP, as R pracma::pchip())",
    "steps": 'steps: each value stays until the next known point (as R approx(method = "constant", f = 0))',
    "nearest": "the value of the nearest known point (a place exactly halfway takes the lower one)",
    "akima": "Akima's curve (Akima 1970, as R akima::aspline())",
    "makima": "the modified Akima curve (as MATLAB makima())",
    "not-a-knot": "a cubic spline with not-a-knot ends (as MATLAB spline())",
    "periodic": 'a periodic cubic spline (as R splinefun(method = "periodic"))',
}
MIN_SMOOTHING_POINTS = 4  # as R smooth.spline()
MAP_LIMIT = 1_000_000  # map_of: places at most


def _example(code: str, name: str = "growth") -> str:
    return f'\nFor example:  {code}   (with sdata.example("{name}"))'


def _plain_choice(value) -> str:
    return str(value).strip().lower().replace("_", " ").replace("neighbors", "neighbours")


def join_choice(step: str, joined_by, directions: int = 1) -> str:
    """joined_by= in its plain form (other names accepted), or a friendly error."""
    choices = JOINED_BY if directions == 1 else JOINED_BY_MANY
    if not isinstance(joined_by, str):
        raise choice_error(step, "joined_by", joined_by, choices)
    choice = _plain_choice(joined_by)
    if choice in ("not a knot", "not-a-knot"):
        choice = "not-a-knot"
    choice = JOIN_ALIASES.get(choice, choice)
    if directions == 1 and choice == "pchip":
        choice = "curve without overshoot"
    if directions > 1 and choice == "smooth curve":
        choice = "smooth surface"
    if choice not in choices:
        if directions == 1 and choice in JOINED_BY_MANY:
            raise SimplydataError(f'joined_by="{choice}" is for two or more directions (at= with two columns). '
                                  f"In one direction use one of: " + ", ".join(f'"{c}"' for c in JOINED_BY))
        if directions > 1 and choice in JOINED_BY:
            raise SimplydataError(f'joined_by="{choice}" is for one direction. With two or more columns in at= '
                                  "use one of: " + ", ".join(f'"{c}"' for c in JOINED_BY_MANY))
        raise choice_error(step, "joined_by", joined_by, choices)
    return choice


def outside_choice(step: str, outside) -> str:
    if not isinstance(outside, str):
        raise choice_error(step, "outside", outside, OUTSIDE)
    choice = _plain_choice(outside)
    choice = OUTSIDE_ALIASES.get(choice, choice)
    if choice not in OUTSIDE:
        raise choice_error(step, "outside", outside, OUTSIDE)
    return choice


def smoothness_choice(step: str, smoothness):
    """0 (through every point), "auto" or a number above 0."""
    if isinstance(smoothness, str):
        if _plain_choice(smoothness) in ("auto", "automatic"):
            return "auto"
    elif (not isinstance(smoothness, bool) and isinstance(smoothness, (int, float, np.integer, np.floating))
          and math.isfinite(smoothness) and smoothness >= 0):
        return float(smoothness)
    raise SimplydataError(
        f'smoothness= should be 0 (the curve passes through every point), "auto" (chosen by cross-validation) '
        f"or a number above 0 (bigger is smoother), not {smoothness!r}."
        + _example(f'table.{step}("population", at={{"year": [2001, 2003]}}, smoothness="auto")')
    )


# ------------------------------------------------------------------ one direction
def unique_mean(x, y):
    """Sorted distinct x, the average y at each, and how many values each had (as R ties = mean)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    uy = np.bincount(inverse, weights=y, minlength=len(ux)) / counts
    return ux, uy, counts


def _band_inverse_trace(m_bands, b_bands):
    """trace(M^-1 B) for symmetric positive pentadiagonal M and B, both given as (3, n) lower bands.

    Only the central five bands of M^-1 are needed; they come from the LDL' factors by the
    recursion of Hutchinson and de Hoog (1985), in O(n).
    """
    from scipy.linalg import cholesky_banded

    n = m_bands.shape[1]
    c = cholesky_banded(m_bands, lower=True)  # c[0] diagonal, c[1] first below, c[2] second below
    d = c[0] ** 2
    low = np.zeros((3, n))
    low[1, :-1] = c[1, :-1] / c[0, :-1]  # L[i+1, i]
    low[2, :-2] = c[2, :-2] / c[0, :-2]  # L[i+2, i]
    inv = np.zeros((3, n))  # inv[k, i] = (M^-1)[i, i + k]

    def get(i, j):
        i, j = min(i, j), max(i, j)
        return inv[j - i, i]

    for i in range(n - 1, -1, -1):
        for j in range(min(i + 2, n - 1), i, -1):
            total = 0.0
            for k in range(1, 3):
                if i + k < n:
                    total -= low[k, i] * get(i + k, j)
            inv[j - i, i] = total
        total = 1.0 / d[i]
        for k in range(1, 3):
            if i + k < n:
                total -= low[k, i] * inv[k, i]
        inv[0, i] = total
    trace = float(np.sum(inv[0] * b_bands[0]))
    trace += 2 * float(np.sum(inv[1, :-1] * b_bands[1, :-1])) + 2 * float(np.sum(inv[2, :-2] * b_bands[2, :-2]))
    return trace


class SmoothingSpline:
    """The natural cubic smoothing spline of R smooth.spline(x, y, all.knots = TRUE).

    x is rescaled to run from 0 to 1, as R does, and the spline g makes
        sum w (y - g(x))**2 + lambda * integral(g''**2)
    as small as possible (Reinsch's algorithm, Green and Silverman 1994, chapter 2), so
    lambda is R's lambda=. Several values at one x are averaged with weight = how many, as
    R. lambda="auto" chooses it by generalised cross-validation exactly as R does:
        GCV = (residual sum of squares / n) / (1 - df / n)**2
    over all n values (ties count their spread around their average), with df the trace of
    the smoother matrix (R's $df). Beyond the ends the curve goes on in a straight line,
    as R predict.smooth.spline.
    """

    def __init__(self, x, y, smoothness="auto"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ux, uy, counts = unique_mean(x, y)
        if len(ux) < 4:
            raise SimplydataError(f"A smoothing spline needs at least 4 different x values; there are {len(ux)}.")
        self.n = len(x)
        self.within = float(np.sum((y - uy[np.searchsorted(ux, x)]) ** 2))
        self.low, self.size = float(ux[0]), float(ux[-1] - ux[0])
        self.x = (ux - self.low) / self.size
        self.y, self.w = uy, counts.astype(float)
        h = np.diff(self.x)
        m = len(self.x)
        # Q (m x m-2) and R (m-2 x m-2) of Green and Silverman, kept as bands
        q0, q1, q2 = 1 / h[:-1], -1 / h[:-1] - 1 / h[1:], 1 / h[1:]  # rows j, j+1, j+2 of column j
        self._q = (q0, q1, q2)
        r_diag = (h[:-1] + h[1:]) / 3
        r_off = h[1:-1] / 6
        self._r = (r_diag, r_off)
        # B = Q' W^-1 Q (pentadiagonal), as lower bands
        winv = 1 / self.w
        b = np.zeros((3, m - 2))
        b[0] = q0 ** 2 * winv[:-2] + q1 ** 2 * winv[1:-1] + q2 ** 2 * winv[2:]
        b[1, :-1] = q1[:-1] * q0[1:] * winv[1:-2] + q2[:-1] * q1[1:] * winv[2:-1]
        b[2, :-2] = q2[:-2] * q0[2:] * winv[2:-2]
        self._b = b
        self._qty = q0 * self.y[:-2] + q1 * self.y[1:-1] + q2 * self.y[2:]
        if smoothness == "auto":
            self.lam = self._gcv_choice()
        else:
            self.lam = float(smoothness)
        self.fitted, self.df = self._fit(self.lam)
        from scipy.interpolate import CubicSpline

        # the natural interpolating spline through the fitted values is the smoothing spline
        self._curve = _natural_ends(CubicSpline(self.x, self.fitted, bc_type="natural"), 0.0, 1.0)

    def _system(self, lam):
        bands = lam * self._b.copy()
        bands[0] += self._r[0]
        bands[1, :-1] += self._r[1]
        return bands

    def _fit(self, lam, want_df=True):
        from scipy.linalg import solveh_banded

        bands = self._system(lam)
        gamma = solveh_banded(bands, self._qty, lower=True)
        q0, q1, q2 = self._q
        qg = np.zeros(len(self.x))
        qg[:-2] += q0 * gamma
        qg[1:-1] += q1 * gamma
        qg[2:] += q2 * gamma
        fitted = self.y - lam * qg / self.w
        df = len(self.x) - lam * _band_inverse_trace(bands, self._b) if want_df else math.nan
        return fitted, df

    def _gcv(self, log_lam):
        lam = 10.0 ** log_lam
        try:
            fitted, df = self._fit(lam)
        except (np.linalg.LinAlgError, ValueError):  # so little smoothing that the sums cannot be done
            return math.inf
        rss = float(np.sum(self.w * (self.y - fitted) ** 2)) + self.within
        return (rss / self.n) / (1 - df / self.n) ** 2

    def _gcv_choice(self):
        from scipy.optimize import minimize_scalar

        grid = np.linspace(-12, 6, 37)
        scores = [self._gcv(g) for g in grid]
        best = int(np.nanargmin(scores))
        left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        found = minimize_scalar(self._gcv, bounds=(left, right), method="bounded", options={"xatol": 1e-7})
        return 10.0 ** float(found.x if found.fun <= scores[best] else grid[best])

    def __call__(self, at):
        return self._curve((np.asarray(at, dtype=float) - self.low) / self.size)


def smoothing_spline(x, y, smoothness="auto"):
    """The smoothing spline through raw (x, y) (ties allowed); see SmoothingSpline."""
    return SmoothingSpline(x, y, smoothness)


def _natural_ends(curve, low, high):
    """Evaluate a curve inside [low, high]; beyond, carry on the end slope in a straight line."""
    slope = curve.derivative()
    at_low, at_high = float(curve(low)), float(curve(high))
    slope_low, slope_high = float(slope(low)), float(slope(high))

    def evaluate(at):
        at = np.asarray(at, dtype=float)
        out = np.asarray(curve(np.clip(at, low, high)), dtype=float)
        out = np.where(at < low, at_low + slope_low * (at - low), out)
        return np.where(at > high, at_high + slope_high * (at - high), out)

    return evaluate


def _curve_through(x, y, how):
    """A function at -> values for the curves through every point (x distinct and sorted)."""
    from scipy import interpolate

    if how == "smooth curve":
        return _natural_ends(interpolate.CubicSpline(x, y, bc_type="natural"), x[0], x[-1])
    if how == "curve without overshoot":
        return interpolate.PchipInterpolator(x, y, extrapolate=True)
    if how in ("akima", "makima"):
        if len(x) < 3:  # SciPy's Akima needs 3 points; through 2 points it is the straight line
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return lambda at: y[0] + slope * (np.asarray(at, dtype=float) - x[0])
        return interpolate.Akima1DInterpolator(x, y, method=how, extrapolate=True)
    if how == "not-a-knot":
        return interpolate.CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)
    if how == "periodic":
        return interpolate.CubicSpline(x, y, bc_type="periodic", extrapolate="periodic")
    raise ValueError(how)  # pragma: no cover


def join_one_direction(x, y, at, how="straight lines", smoothness=0, outside="empty"):
    """Values at `at` on the curve through (x, y). x must be distinct and sorted (see unique_mean),
    except for the smoothing spline (smoothness= not 0), which takes the raw points.

    Gives back the values (empty, NaN, outside when outside="empty").
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    at = np.asarray(at, dtype=float)
    low, high = np.min(x), np.max(x)
    place = np.clip(at, low, high) if outside == "nearest end" else at
    if isinstance(smoothness, SmoothingSpline):  # already fitted
        values = smoothness(place)
    elif smoothness != 0:
        values = smoothing_spline(x, y, smoothness)(place)
    elif how == "straight lines":
        values = np.interp(place, x, y)
    elif how == "steps":
        index = np.clip(np.searchsorted(x, place, side="right") - 1, 0, len(x) - 1)
        values = y[index]
    elif how == "nearest":
        right = np.clip(np.searchsorted(x, place, side="left"), 0, len(x) - 1)
        left = np.clip(right - 1, 0, len(x) - 1)
        take_left = np.abs(place - x[left]) <= np.abs(x[right] - place)
        values = np.where(take_left, y[left], y[right])
    else:
        values = np.asarray(_curve_through(x, y, how)(place), dtype=float)
    values = np.array(values, dtype=float)
    if outside == "empty":
        values[(at < low) | (at > high)] = np.nan
    values[np.isnan(at)] = np.nan
    return values


def _need_points(how, smoothness, distinct, step, group_words=""):
    """The fewest distinct places a way of joining needs; friendly error when there are fewer."""
    need = 2
    if smoothness != 0:
        need = MIN_SMOOTHING_POINTS
    elif how == "periodic":
        need = 3
    if distinct < need:
        what = ("a smoothing spline" if smoothness != 0 else f'joined_by="{how}"')
        raise SimplydataError(f"{what} needs at least {need} different known places{group_words}; "
                              f"there {'is' if distinct == 1 else 'are'} {distinct}.")


# ------------------------------------------------------------------ two or more directions
def full_grid(points):
    """The distinct values along each direction if the points form a full grid, else None."""
    axes = [np.unique(points[:, k]) for k in range(points.shape[1])]
    if len(points) != math.prod(len(a) for a in axes):
        return None
    if any(len(a) < 2 for a in axes):
        return None
    return axes


def _grid_values(points, values, axes):
    index = [np.searchsorted(a, points[:, k]) for k, a in enumerate(axes)]
    grid = np.full([len(a) for a in axes], np.nan)
    grid[tuple(index)] = values
    return grid


def unique_points(points, values):
    """Distinct points and the average value at each (and how many places had several)."""
    unique, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    means = np.bincount(inverse, weights=values, minlength=len(unique)) / counts
    return unique, means, int((counts > 1).sum())


def join_many_directions(points, values, at, how="straight lines", smoothness=0.0, outside="empty"):
    """Values at the rows of `at` from scattered or gridded points (distinct points, see unique_points).

    Gives back (values, kind) where kind is "grid" or "scattered".
    """
    from scipy import interpolate, spatial

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    at = np.asarray(at, dtype=float)
    axes = full_grid(points)
    kind = "grid" if axes is not None else "scattered"
    if how in ("cubic", "quintic", "pchip") and axes is None:
        raise SimplydataError(f'joined_by="{how}" needs points on a full grid (every value of one column with every '
                              'value of the others). These points are scattered: use joined_by="smooth surface" '
                              'or "straight lines".')
    if smoothness != 0 and how != "smooth surface":
        raise SimplydataError('smoothness= with two or more directions goes with joined_by="smooth surface".')
    if how in ("smooth surface",):
        surface = interpolate.RBFInterpolator(points, values, kernel="thin_plate_spline",
                                              smoothing=float(smoothness))
        out = surface(at)
        if outside != "extend":
            inside = _inside(points, at, axes)
            if outside == "empty":
                out[~inside] = np.nan
            else:
                near = interpolate.NearestNDInterpolator(points, values)(at)
                out[~inside] = near[~inside] if axes is None else surface(_clip_to(at, axes))[~inside]
        return out, kind
    if axes is not None:
        grid = _grid_values(points, values, axes)
        method = "linear" if how == "straight lines" else how
        if outside == "extend":
            if how in ("straight lines", "nearest"):
                raise SimplydataError(f'outside="extend" carries a curve or surface on beyond the points; with '
                                      f'joined_by="{how}" use outside="nearest end" or outside="empty".')
            reader = interpolate.RegularGridInterpolator(axes, grid, method=method, bounds_error=False,
                                                         fill_value=None)
            return reader(at), kind
        reader = interpolate.RegularGridInterpolator(axes, grid, method=method, bounds_error=False,
                                                     fill_value=np.nan)
        place = _clip_to(at, axes) if outside == "nearest end" else at
        return reader(place), kind
    if outside == "extend":
        raise SimplydataError(f'outside="extend" needs joined_by="smooth surface" for scattered points; flat '
                              'triangles and nearest points cannot be carried on. Use outside="nearest end".')
    if how == "nearest":
        out = interpolate.NearestNDInterpolator(points, values)(at)
        if outside == "empty":
            out[~_inside(points, at, None)] = np.nan
        return out, kind
    try:
        triangles = spatial.Delaunay(points)
    except Exception:
        raise SimplydataError("The points lie on one line (or plane), so no triangles can be made between them. "
                              'Use joined_by="nearest", or one direction only.') from None
    out = interpolate.LinearNDInterpolator(triangles, values)(at)
    if outside == "nearest end":
        empty = np.isnan(out)
        out[empty] = interpolate.NearestNDInterpolator(points, values)(at[empty])
    return out, kind


def _clip_to(at, axes):
    return np.column_stack([np.clip(at[:, k], a[0], a[-1]) for k, a in enumerate(axes)])


def _inside(points, at, axes):
    """True for places inside the grid's box, or inside the outline (convex hull) of scattered points."""
    if axes is not None:
        return np.all([(at[:, k] >= a[0]) & (at[:, k] <= a[-1]) for k, a in enumerate(axes)], axis=0)
    from scipy import spatial

    try:
        return spatial.Delaunay(points).find_simplex(at) >= 0
    except Exception:
        return np.zeros(len(at), dtype=bool)


# ------------------------------------------------------------------ fill_empty_cells(with_the="curve ...")
def curve_between_neighbours(values, x=None, groups=None, how="smooth curve"):
    """Fill empty values (NaN) on a curve through the filled ones, as line_between_neighbours.

    how="smooth curve": natural cubic spline (R zoo::na.spline(method = "natural"));
    how="curve without overshoot": PCHIP (MATLAB fillmissing(..., "pchip")).
    Values before the first or after the last filled value of a group stay empty.
    """
    values = np.asarray(values, dtype=float)
    out = values.copy()
    x = np.arange(len(values), dtype=float) if x is None else np.asarray(x, dtype=float)
    codes = np.zeros(len(values), dtype=int) if groups is None else np.asarray(groups)
    for code in np.unique(codes):
        where = np.flatnonzero(codes == code)
        xs, ys = x[where], values[where]
        known = ~np.isnan(ys) & ~np.isnan(xs)
        if not known.any():
            continue
        kx, ky, _ = unique_mean(xs[known], ys[known])
        wanted = np.isnan(ys) & ~np.isnan(xs) & (xs >= kx[0]) & (xs <= kx[-1])
        if not wanted.any():
            continue
        if len(kx) < 2:  # all filled cells at one place: the gap sits on that place
            out[where[wanted]] = ky[0]
            continue
        out[where[wanted]] = join_one_direction(kx, ky, xs[wanted], how=how)
    return out


# ------------------------------------------------------------------ the table words
def _as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if hasattr(value, "to_list"):
        return list(value.to_list())
    return [value]


def _is_number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float, np.integer, np.floating))


DAY = np.timedelta64(86_400_000_000, "us")


def _days(values) -> np.ndarray:
    """Dates as days since 1970 (floats); empty dates become NaN."""
    raw = np.asarray(values).astype("datetime64[us]")
    out = (raw - np.datetime64("1970-01-01", "us")) / DAY
    out = np.asarray(out, dtype=float)
    out[np.isnat(raw)] = np.nan
    return out


def _dates_from_days(days) -> np.ndarray:
    days = np.asarray(days, dtype=float)
    micro = np.round(days * 86_400_000_000)
    out = np.full(len(days), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    ok = np.isfinite(days)
    out[ok] = np.datetime64("1970-01-01", "us") + micro[ok].astype("int64").astype("timedelta64[us]")
    return out


class BetweenWords:
    """values_at() and map_of(): values between known points (see the module notes)."""

    # ------------------------------------------------------------ helpers
    def _direction_values(self, step, name, example):
        """A direction column as floats (dates as days) and whether it holds dates."""
        kind = self._kind_of(name)
        series = self._frame[name]
        if kind == "date":
            return _days(series.to_numpy()), True
        if kind != "number":
            raise SimplydataError(f'{step}() needs numbers (or dates) in "{name}" to place the points, but it is a '
                                  f"{kind} column.{example}")
        return series.cast(nw.Float64).to_numpy().astype(float), False

    @staticmethod
    def _wanted_places(step, name, given, dates, example):
        """The places asked for in at= for one column, as floats (dates as days)."""
        values = _as_list(given)
        if not values:
            raise SimplydataError(f'at= has no places for "{name}".{example}')
        if dates:
            import pandas as pd

            try:
                stamps = pd.to_datetime(pd.Series(values))
            except (ValueError, TypeError):
                raise SimplydataError(f'"{name}" holds dates, so at= needs dates for it, such as "2026-03-01".'
                                      f"{example}") from None
            return _days(stamps.to_numpy())
        bad = [v for v in values if not _is_number(v) or not math.isfinite(v)]
        if bad:
            raise SimplydataError(f'at= needs numbers for "{name}", not {bad[0]!r}.{example}')
        return np.asarray(values, dtype=float)

    def _between_group_keys(self, step, for_each, used, example):
        if for_each is None:
            return []
        keys = [for_each] if isinstance(for_each, str) else _as_list(for_each)
        if not keys or not all(isinstance(k, str) for k in keys):
            raise SimplydataError(f'for_each= should be a column name, such as for_each="country".{example}')
        self._need_columns(*keys)
        clash = [k for k in keys if k in used]
        if clash:
            raise SimplydataError(f'"{clash[0]}" cannot be both in at= (or the column) and in for_each=.{example}')
        return keys

    def _between_frame(self, step, column, directions, keys, example):
        """Numbers of the column and the directions, group labels, with empty rows skipped (noted)."""
        kind = self._kind_of(column)
        if kind != "number":
            raise SimplydataError(f'{step}() estimates a column of numbers, but "{column}" is a {kind} column.'
                                  f"{example}")
        y = self._frame[column].cast(nw.Float64).to_numpy().astype(float)
        places, dates = [], []
        for name in directions:
            values, is_date = self._direction_values(step, name, example)
            places.append(values)
            dates.append(is_date)
        points = np.column_stack(places) if places else np.empty((len(y), 0))
        labels = [tuple(self._frame[k].to_list()[i] for k in keys) for i in range(len(y))] if keys else None
        keep = ~np.isnan(y) & ~np.isnan(points).any(axis=1)
        if keys:
            keep &= np.array([all(v is not None and v == v for v in lab) for lab in labels])
        _skipped_note(int((~keep).sum()), [column, *directions, *keys])
        if labels is not None:
            labels = [lab for lab, k in zip(labels, keep) if k]
        return y[keep], points[keep], labels, dates

    # ------------------------------------------------------------ the engine for both words
    def _estimate(self, step, column, directions, at_places, how, smoothness, outside, keys, example):
        """Estimates for every group at the rows of at_places (a 2-D array). Prints the notes.

        Gives back (list of (group label, estimates)), facts for the sentence.
        """
        y, points, labels, dates = self._between_frame(step, column, directions, keys, example)
        if len(y) == 0:
            raise SimplydataError(f'"{column}" has no numbers in rows where {_list_words(f"{chr(34)}{d}{chr(34)}" for d in directions)} '
                                  "are filled, so there is nothing to join.")
        if labels is None:
            groups = [(None, np.ones(len(y), dtype=bool))]
        else:
            order = []
            for lab in labels:
                if lab not in order:
                    order.append(lab)
            try:
                order = sorted(order)
            except TypeError:
                pass
            codes = np.array([order.index(lab) for lab in labels])
            groups = [(lab, codes == i) for i, lab in enumerate(order)]
        results, facts = [], {"outside": 0, "ties": 0, "kind": None, "ranges": [], "too_few": [], "spline": []}
        for label, rows in groups:
            words = "" if label is None else f" in the group {', '.join(str(v) for v in label)}"
            if len(directions) == 1:
                x, ys, counts = unique_mean(points[rows, 0], y[rows])
                facts["ties"] += int((counts > 1).sum())
                try:
                    _need_points(how, smoothness, len(x), step, words)
                except SimplydataError:
                    if label is None:
                        raise
                    facts["too_few"].append(label)
                    results.append((label, np.full(len(at_places), np.nan)))
                    continue
                if how == "periodic" and not math.isclose(ys[0], ys[-1], rel_tol=1e-9, abs_tol=1e-12):
                    raise SimplydataError(
                        f'joined_by="periodic" is for a cycle, so the first and last known values must be the same; '
                        f"here they are {_num(ys[0])} and {_num(ys[-1])}{words}. Repeat the first value one period "
                        "later, or use joined_by=\"smooth curve\".")
                place = at_places[:, 0]
                if smoothness != 0:  # the smoothing spline takes every point (R counts ties in its GCV)
                    spline = smoothing_spline(points[rows, 0], y[rows], smoothness)
                    facts["spline"].append(spline)
                    estimates = join_one_direction(points[rows, 0], y[rows], place, how, spline, outside)
                else:
                    estimates = join_one_direction(x, ys, place, how, smoothness, outside)
                out = (place < x[0]) | (place > x[-1])
                facts["ranges"].append((x[0], x[-1]))
            else:
                pts, vals, ties = unique_points(points[rows], y[rows])
                facts["ties"] += ties
                if len(pts) < len(directions) + 1:
                    if label is None:
                        raise SimplydataError(f"{step}() needs at least {len(directions) + 1} different known points "
                                              f"to join in {len(directions)} directions; there are {len(pts)}.")
                    facts["too_few"].append(label)
                    results.append((label, np.full(len(at_places), np.nan)))
                    continue
                estimates, kind = join_many_directions(pts, vals, at_places, how, smoothness, outside)
                facts["kind"] = kind
                out = ~_inside(pts, at_places, full_grid(pts))
            facts["outside"] += int(out.sum())
            results.append((label, estimates))
        facts["dates"] = dates
        return results, facts

    def _say_method(self, how, smoothness, facts, directions):
        if smoothness != 0 and len(directions) == 1:
            splines = facts.get("spline") or []
            size = ""
            if len(splines) == 1:
                size = f", {splines[0].df:.2f} equivalent degrees of freedom"
            if smoothness == "auto":
                chosen = f"lambda {splines[0].lam:.3g}, " if len(splines) == 1 else "its smoothness "
                return ("a smoothing spline that passes near, not through, the known points (" + chosen +
                        f"chosen by generalised cross-validation{size}; as R smooth.spline(x, y, all.knots = TRUE))")
            return (f"a smoothing spline that passes near, not through, the known points ({size[2:] + '; ' if size else ''}"
                    f"as R smooth.spline(x, y, all.knots = TRUE, lambda = {smoothness:g}))")
        if len(directions) == 1:
            return SAID[how]
        if how == "straight lines":
            if facts["kind"] == "grid":
                return ("straight lines along each direction, because the points form a full grid "
                        "(bilinear interpolation)")
            return ("flat triangles between the points (linear interpolation on a Delaunay triangulation, "
                    "as R interp::interp(linear = TRUE))")
        if how == "smooth surface":
            extra = f", smoothness {_num(smoothness)}" if smoothness else ", passing through every point"
            return f"a smooth surface (a thin-plate spline{extra})"
        if how == "nearest":
            return "the value of the nearest known point"
        return f'{how} curves along each direction of the full grid (SciPy RegularGridInterpolator(method="{how}"))'

    def _notes(self, facts, directions, outside, how, smoothness, count, keys):
        if facts["ties"]:
            n = facts["ties"]
            where = "place" if n == 1 else "places"
            print(f"Note: {n} {where} had several known values; their average was used (as R approx(ties = mean)).")
        if facts["too_few"]:
            names = _list_words(", ".join(str(v) for v in lab) for lab in facts["too_few"])
            print(f"Note: {names} had too few known points to join, so {'its' if len(facts['too_few']) == 1 else 'their'}"
                  " values are empty.")
        n = facts["outside"]
        if not n:
            return
        places = "place was" if n == 1 else "places were"
        if len(directions) == 1:
            if len(facts["ranges"]) == 1 and not facts["dates"][0]:
                low, high = facts["ranges"][0]
                beyond = f'outside {_num(low)} to {_num(high)} in "{directions[0]}"'
            else:
                beyond = f'outside the known range of "{directions[0]}"' + (" in their group" if keys else "")
        elif facts["kind"] == "grid":
            beyond = "outside the grid of known points"
        else:
            beyond = "outside the outline of the known points"
        if outside == "empty":
            fill = ('outside="nearest end" fills them' if how in ("straight lines", "steps", "nearest")
                    and smoothness == 0 else 'outside="nearest end" or outside="extend" fills them')
            print(f"Note: {n} {places} {beyond}, so {'it is' if n == 1 else 'they are'} empty; {fill}.")
        elif outside == "nearest end":
            what = ("the value at the nearest end" if len(directions) == 1 or facts["kind"] == "grid"
                    else "the value of the nearest known point")
            print(f"Note: {n} {places} {beyond}; {'it has' if n == 1 else 'they have'} {what}.")
        else:
            print(f"Note: {n} {places} {beyond}; the curve was carried on there, which is guesswork: "
                  "nothing was measured beyond the known points.")

    def _check_extend(self, how, smoothness, outside, directions):
        if outside == "extend" and len(directions) == 1 and smoothness == 0 and how not in CURVES:
            raise SimplydataError(
                f'outside="extend" carries a curve on beyond the known points; with joined_by="{how}" use '
                'outside="nearest end" (the end value) or outside="empty", or join by a curve such as '
                'joined_by="smooth curve".')

    def _result(self, directions, at_places, dates, keys, results, column):
        """The new table: the at= columns, the group columns, and the estimated column."""
        backend = self._frame.implementation
        n_at = len(at_places)
        data = {}
        for k, name in enumerate(directions):
            place = np.tile(at_places[:, k], len(results))
            data[name] = _dates_from_days(place) if dates[k] else place
        for j, key in enumerate(keys):
            data[key] = [label[j] for label, _ in results for _i in range(n_at)]
        data[column] = np.concatenate([est for _, est in results]) if results else np.array([])
        series = []
        for name, values in data.items():
            if isinstance(values, np.ndarray) and values.dtype.kind == "M":
                series.append(nw.new_series(name, values, nw.Datetime("us"), backend=backend))
            elif isinstance(values, np.ndarray):
                dtype = self._frame.schema[name] if name != column else nw.Float64
                whole = dtype.is_integer() and len(values) and np.all(np.isfinite(values)) \
                    and np.all(values == np.round(values))
                if whole:  # whole years stay whole numbers
                    series.append(nw.new_series(name, values.astype("int64"), nw.Int64, backend=backend))
                else:
                    series.append(nw.new_series(name, values.astype(float), nw.Float64, backend=backend))
            else:
                series.append(nw.new_series(name, values, self._frame.schema[name], backend=backend))
        frame = nw.from_dict({s.name: s for s in series}, backend=backend)
        return self._new_table(frame)

    # ------------------------------------------------------------ values_at
    @friendly
    def values_at(self, column=None, at=None, joined_by="straight lines", smoothness=0, for_each=None,
                  outside="empty"):
        """Estimate a column at places between the known points, by joining the dots.

        table.values_at("population", at={"year": [2001, 2003, 2011]}, for_each="country")   (with sdata.example("growth"))
        at= names the column(s) that place the points and the places wanted. One column joins
        the points along a line; two or more (at={"east": [1.5, 2.5], "north": [3, 4.5]},
        sdata.example("island")) join them across a surface: the lists are read as pairs,
        (1.5, 3) and (2.5, 4.5); table.map_of() gives every combination.
        joined_by= says how the points are joined (one column): "straight lines" (default,
        linear interpolation, R approx()), "smooth curve" (natural cubic spline, R
        splinefun(method = "natural")), "curve without overshoot" (PCHIP), "steps", "nearest";
        experts: "akima", "makima", "not-a-knot", "periodic". Two or more columns: "straight
        lines" (bilinear on a full grid, triangles otherwise), "smooth surface" (thin-plate
        spline), "nearest"; on a full grid also "cubic", "quintic", "pchip".
        smoothness="auto" or a number gives a smoothing spline that passes near, not through,
        the points (R smooth.spline(); a number is R's lambda=); 0 passes through every point.
        outside= is what happens beyond the known points: "empty" (default), "nearest end" or
        "extend" (curves only; guesswork). for_each="country" joins each group on its own.
        Rows with an empty cell are skipped. Gives back a new table: the at= columns, the
        for_each= columns and the estimated column.
        """
        step = "values_at"
        example = _example('table.values_at("population", at={"year": [2001, 2003]}, for_each="country")')
        if not isinstance(column, str):
            raise SimplydataError(f"values_at() needs the column to estimate, in quotes.{example}")
        self._need_columns(column)
        if not isinstance(at, dict) or not at:
            raise SimplydataError("values_at() needs at= with the column that places the points and the places "
                                  f'wanted, such as at={{"year": [2001, 2003]}}.{example}')
        directions = list(at)
        if not all(isinstance(d, str) for d in directions):
            raise SimplydataError(f"at= should use column names as keys.{example}")
        self._need_columns(*directions)
        if column in directions:
            raise SimplydataError(f'"{column}" is the column to estimate, so it cannot also be in at=.{example}')
        how = join_choice(step, joined_by, len(directions))
        smoothness = smoothness_choice(step, smoothness)
        outside = outside_choice(step, outside)
        self._check_smoothness(how, smoothness, directions)
        self._check_extend(how, smoothness, outside, directions)
        keys = self._between_group_keys(step, for_each, [column, *directions], example)
        dates = [self._kind_of(d) == "date" for d in directions]
        places = [self._wanted_places(step, d, at[d], is_date, example) for d, is_date in zip(directions, dates)]
        sizes = {len(p) for p in places if len(p) != 1}
        if len(sizes) > 1:
            lengths = ", ".join(f'{len(p)} for "{d}"' for d, p in zip(directions, places))
            raise SimplydataError(f"at= gives places as pairs, so each list needs the same length ({lengths}). "
                                  "For every combination of the values use table.map_of(...)."
                                  + _example('table.values_at("height", at={"east": [1.5, 2.5], "north": [3, 4.5]})',
                                             "island"))
        n = max(len(p) for p in places)
        at_places = np.column_stack([np.repeat(p, n) if len(p) == 1 else p for p in places])
        results, facts = self._estimate(step, column, directions, at_places, how, smoothness, outside, keys, example)
        groups = (f' for each of the {len(results)} groups in {_list_words(f"{chr(34)}{k}{chr(34)}" for k in keys)}'
                  if keys else "")
        what = "value" if n == 1 else "values"
        where = (f'{n} {what} of "{directions[0]}"' if len(directions) == 1
                 else f"{n} {'place' if n == 1 else 'places'} of "
                 + _list_words(f'"{d}"' for d in directions))
        print(f'Estimated "{column}" at {where}{groups}, joined by '
              f"{self._say_method(how, smoothness, facts, directions)}.")
        self._notes(facts, directions, outside, how, smoothness, n, keys)
        return self._result(directions, at_places, dates, keys, results, column)

    @staticmethod
    def _check_smoothness(how, smoothness, directions):
        if smoothness == 0:
            return
        if len(directions) == 1 and how not in ("straight lines", "smooth curve"):
            raise SimplydataError(
                f'smoothness= gives a smoothing spline, which is its own way of joining; leave out '
                f'joined_by="{how}" (or use joined_by="smooth curve").')
        if len(directions) > 1:
            if how != "smooth surface":
                raise SimplydataError('smoothness= with two or more directions goes with joined_by="smooth surface".')
            if smoothness == "auto":
                raise SimplydataError('smoothness="auto" works in one direction. For a surface give a number, such as '
                                      'smoothness=0.1 (bigger is smoother).')

    # ------------------------------------------------------------ map_of
    @friendly
    def map_of(self, column=None, across=None, steps=50, joined_by="straight lines", smoothness=0):
        """Estimate a column at evenly spaced places across one or more columns (a map), ready for a contour or heatmap.

        table.map_of("height", across=["east", "north"], steps=100)   (with sdata.example("island"))
        The map runs from the smallest to the biggest known value of each across= column,
        with steps= places along each (a number, or one per column). The points are joined
        as in table.values_at() (same joined_by= and smoothness=). Places outside the outline
        of the known points stay empty, with a note. Gives back a new long table: one row per
        place, with the across= columns and the estimated column.
        """
        step = "map_of"
        example = _example('table.map_of("height", across=["east", "north"], steps=100)', "island")
        if not isinstance(column, str):
            raise SimplydataError(f"map_of() needs the column to estimate, in quotes.{example}")
        self._need_columns(column)
        if across is None:
            raise SimplydataError(f"map_of() needs across= with the columns that place the points.{example}")
        directions = [across] if isinstance(across, str) else _as_list(across)
        if not directions or not all(isinstance(d, str) for d in directions):
            raise SimplydataError(f"across= should be a column name or a list of column names.{example}")
        self._need_columns(*directions)
        if column in directions:
            raise SimplydataError(f'"{column}" is the column to estimate, so it cannot also be in across=.{example}')
        counts = _as_list(steps)
        if len(counts) == 1:
            counts = counts * len(directions)
        if len(counts) != len(directions) or not all(
                not isinstance(c, bool) and isinstance(c, (int, np.integer)) and c >= 2 for c in counts):
            raise SimplydataError(f"steps= should be a whole number from 2 up (or one per across= column), "
                                  f"not {steps!r}.{example}")
        total = math.prod(int(c) for c in counts)
        if total > MAP_LIMIT:
            raise SimplydataError(f"That map would have {total:,} places; the most is {MAP_LIMIT:,}. "
                                  "Use fewer steps=.")
        how = join_choice(step, joined_by, len(directions))
        smoothness = smoothness_choice(step, smoothness)
        self._check_smoothness(how, smoothness, directions)
        y, points, _, dates = self._between_frame(step, column, directions, [], example)
        if len(y) == 0:
            raise SimplydataError(f'"{column}" has no numbers to join.')
        lines = [np.linspace(np.nanmin(points[:, k]), np.nanmax(points[:, k]), int(c))
                 for k, c in enumerate(counts)]
        mesh = np.meshgrid(*lines, indexing="ij")
        at_places = np.column_stack([m.ravel() for m in mesh])
        results, facts = self._estimate(step, column, directions, at_places, how, smoothness, "empty", [], example)
        size = " x ".join(str(int(c)) for c in counts)
        print(f'Estimated "{column}" at {size} evenly spaced places across {_list_words(f"{chr(34)}{d}{chr(34)}" for d in directions)}, '
              f"joined by {self._say_method(how, smoothness, facts, directions)}.")
        estimates = results[0][1]
        empty = int(np.isnan(estimates).sum())
        if empty:
            print(f"Note: {empty} {'place is' if empty == 1 else 'places are'} outside the outline of the known "
                  "points, so they are empty.")
        if facts["ties"]:
            print(f"Note: {facts['ties']} places had several known values; their average was used.")
        return self._result(directions, at_places, dates, [], results, column)
