"""Statistics and science extras: error bars, fit lines, reference lines, shading.

These words work on the most recent data layer (ctx.last) and are drawn in the "extra"
phase, after every chart type. Methods, written so they can be stated in a paper:

* error_bars: for every x value (and colour group) the mean of y, with
  "sd" = sample standard deviation (n - 1), "se" = sd / sqrt(n),
  "ci" = 95 % confidence interval of the mean from the t distribution (_stats.t_interval).
  Empty cells are skipped. An x value with a single y value gets no bar.
  With colour groups sharing an x value, each group's bars, line and dots are shifted
  sideways together (ggplot2's position_dodge), spread over 15 % of the smallest x gap
  (15 % of 1 for categories); dodge=False keeps the true x.
* fit_line: ordinary least squares, scipy.stats.linregress; R² = rvalue².
  band=True is the 95 % confidence band of the fitted line (of the mean of y at each x):
  y_hat ± t(0.975, n - 2) · s · sqrt(1/n + (x - x̄)² / Sxx), with s² = SSE / (n - 2).
* fit_curve: least-squares polynomial, numpy.polyfit; R² = 1 - SSE / SST.
* smooth_line: LOWESS (Cleveland 1979), the algorithm of R's lowess() and statsmodels'
  lowess(): local straight-line fits with tricube weights over the nearest 2/3 of the
  points, then 3 robustness iterations with bisquare weights (6 × median absolute
  residual). Ported to NumPy here so statsmodels is not needed. Every point is fitted
  (delta = 0, as in statsmodels); above 1000 points, points within 1 % of the x range
  are interpolated (delta = 0.01 · range, R's default), to keep it fast.
* side_histograms: numpy.histogram_bin_edges(bins="auto"), the same bins for all groups.
* ps.pairs: seaborn.histplot on the diagonal, or seaborn.kdeplot (Gaussian kernel,
  Scott's bandwidth) when colour= makes groups; seaborn.scatterplot elsewhere.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.collections import PathCollection
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.figure import Figure
from scipy import stats

from .. import _style
from .._core import PHASES, BaseChart
from .._data import is_listlike
from .._errors import PlotsimpleError, column_error, friendly
from .._stats import ERRORS, t_interval
from ._helpers import colour_role, default_colour, palette_for
from .look import LookWords
from .text import TextWords

try:  # the order seaborn gives to groups, so our colours match the dots
    from seaborn._base import categorical_order as _sns_order
except ImportError:  # pragma: no cover
    _sns_order = None

# Chart types that make a data layer. Used to spot a wrong layer at call time.
_CHART_KINDS = {
    "bars", "lines", "dots", "area", "histogram", "boxes", "violins", "heatmap", "pie",
    "polar", "contour", "surface", "dots_3d", "density", "count_bars", "all_points",
}
_EXAMPLE = {
    "dots": 'chart.dots(x="height", y="weight")',
    "lines": 'chart.lines(x="day", y="temperature")',
}


# ====================================================================== calculations
def group_errors(table, x, y, group=None, kind="sd") -> pd.DataFrame:
    """Mean and error half-width of y for every x value (and colour group).

    Returns columns x, group (if any), mean, n, half. Empty cells in y are skipped.
    half is NaN for groups with fewer than 2 values.
    """
    keys = [x] + ([group] if group else [])
    df = table[keys + [y]].dropna(subset=[y])
    rows = []
    for key, part in df.groupby(keys, sort=False, dropna=True):
        v = part[y].to_numpy(dtype=float)
        n = len(v)
        mean = v.mean()
        if n < 2:
            half = np.nan
        elif kind == "sd":
            half = v.std(ddof=1)
        elif kind == "se":
            half = v.std(ddof=1) / math.sqrt(n)
        else:  # "ci"
            low, high = t_interval(v)
            half = (high - low) / 2
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, key), mean=mean, n=n, half=half))
    return pd.DataFrame(rows, columns=keys + ["mean", "n", "half"])


def linear_fit(x, y) -> dict:
    """Straight line by least squares (scipy.stats.linregress) plus what the band needs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.linregress(x, y)
    n = len(x)
    fit = dict(slope=res.slope, intercept=res.intercept, r2=res.rvalue ** 2, n=n,
               x_mean=x.mean(), sxx=float(((x - x.mean()) ** 2).sum()))
    if n > 2:
        resid = y - (res.intercept + res.slope * x)
        fit["s"] = math.sqrt((resid ** 2).sum() / (n - 2))
        fit["t"] = stats.t.ppf(0.975, n - 2)
    return fit


def line_band(fit: dict, xs):
    """95 % confidence band of the fitted line at xs: (low, high). Needs n >= 3."""
    xs = np.asarray(xs, dtype=float)
    centre = fit["intercept"] + fit["slope"] * xs
    half = fit["t"] * fit["s"] * np.sqrt(1 / fit["n"] + (xs - fit["x_mean"]) ** 2 / fit["sxx"])
    return centre - half, centre + half


def poly_fit(x, y, degree: int) -> dict:
    """Least-squares polynomial (numpy.polyfit). coefs are highest power first."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coefs = np.polyfit(x, y, degree)
    pred = np.polyval(coefs, x)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / sst if sst > 0 else np.nan
    return dict(coefs=coefs, r2=r2, n=len(x))


def lowess(x, y, frac: float = 2 / 3, iterations: int = 3, delta: float = 0.0):
    """LOWESS smoother (Cleveland 1979), a NumPy port of R's clowess / statsmodels' lowess.

    Returns (x_sorted, fitted) with x sorted ascending. frac is the share of points in
    each local fit, iterations the number of robustness steps, delta the distance within
    which points are interpolated instead of fitted (0 = fit every point).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    n = len(x)
    if n < 2:
        return x, y.copy()
    ns = max(min(int(frac * n + 1e-7), n), 2)
    xrange = x[-1] - x[0]
    fitted = np.empty(n)
    rw = np.ones(n)

    def local(xs, nleft, nright, robust):
        h = max(xs - x[nleft], x[nright] - xs)
        h9, h1 = 0.999 * h, 0.001 * h
        # from nleft up to the last point within h9 on the right (so ties are included)
        end = max(int(np.searchsorted(x, xs + h9, side="right")), nleft + 1)
        xj = x[nleft:end]
        r = np.abs(xj - xs)
        if h > 0:
            w = np.where(r <= h9, np.where(r <= h1, 1.0, (1 - (r / h) ** 3) ** 3), 0.0)
        else:
            w = (r <= 0).astype(float)
        if robust:
            w = w * rw[nleft:end]
        a = w.sum()
        if a <= 0:
            return None
        w = w / a
        if h > 0:  # local straight line instead of a local average
            a = (w * xj).sum()
            b = xs - a
            c = (w * (xj - a) ** 2).sum()
            if math.sqrt(c) > 0.001 * xrange:
                b /= c
                w = w * (b * (xj - a) + 1)
        return float((w * y[nleft:end]).sum())

    for it in range(iterations + 1):
        nleft, nright = 0, ns - 1
        last = -1
        i = 0
        while True:
            while nright < n - 1:  # slide the window of ns nearest points
                if x[i] - x[nleft] <= x[nright + 1] - x[i]:
                    break
                nleft += 1
                nright += 1
            value = local(x[i], nleft, nright, it > 0)
            fitted[i] = y[i] if value is None else value
            if last < i - 1:  # points skipped because of delta: straight-line interpolation
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    fitted[j] = alpha * fitted[i] + (1 - alpha) * fitted[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    fitted[i] = fitted[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        if it == iterations:
            break
        res = np.abs(y - fitted)
        sc = res.sum() / n
        cmad = 6 * np.median(res)
        if cmad < 1e-7 * sc:
            break
        c9, c1 = 0.999 * cmad, 0.001 * cmad
        rw = np.where(res <= c1, 1.0, np.where(res <= c9, (1 - (res / cmad) ** 2) ** 2, 0.0))
    return x, fitted


# ====================================================================== text helpers
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_MINUS = "−"


def nice_number(v: float) -> str:
    """About 3 significant figures, without scientific notation for everyday sizes."""
    if v == 0:
        return "0"
    if not np.isfinite(v):
        return str(v)
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        text = f"{a:.3g}"
    else:
        digits = max(0, 2 - math.floor(math.log10(a)))
        text = f"{a:.{digits}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    return (_MINUS if v < 0 else "") + text


def equation_text(coefs, r2) -> str:
    """'y = 2.31x + 0.48, R² = 0.94' from coefficients, highest power first."""
    degree = len(coefs) - 1
    parts = []
    for k, c in enumerate(coefs):
        power = degree - k
        num = nice_number(abs(c))
        if power == 0:
            term = num
        else:
            term = ("" if num == "1" else num) + "x" + (str(power).translate(_SUPERSCRIPT) if power > 1 else "")
        if not parts:
            parts.append((_MINUS if c < 0 else "") + term)
        else:
            parts.append((f" {_MINUS} " if c < 0 else " + ") + term)
    r2_text = f"{r2:.2f}" if np.isfinite(r2) else "?"
    return "y = " + "".join(parts) + f", R² = {r2_text}"


def _write_equation(ctx, text, colour):
    """Equations stack in the top-left corner, one line each."""
    count = ctx.store.get("equations", 0)
    ctx.store["equations"] = count + 1
    size = rcParams["legend.fontsize"]
    ctx.ax.annotate(text, xy=(0.02, 0.98), xycoords="axes fraction",
                    xytext=(0, -count * 1.6 * size), textcoords="offset points",
                    ha="left", va="top", color=colour, fontsize=size, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=rcParams["axes.facecolor"],
                              edgecolor="none", alpha=0.75))


# ====================================================================== layer helpers
def _wrong_layer(word, kinds, found):
    names = " or ".join(sorted(kinds))
    example = _EXAMPLE[sorted(kinds)[0]]
    if found is None:
        return PlotsimpleError(f".{word}() works on {names}, but your chart has none yet.\n"
                               f"Add one first, for example {example}")
    return PlotsimpleError(f".{word}() works on {names}, but the last chart type is {found}.\n"
                           f"Use it after {example}")


def _layer(ctx, word, kinds):
    """The data layer this word works on, or a friendly error."""
    last = ctx.last
    if not last or last.get("kind") not in kinds:
        raise _wrong_layer(word, kinds, last.get("kind") if last else None)
    return last


def _groups(ctx, table, group, fixed=None):
    """Yield (name, part of table, colour) for every colour group, matching the dots' colours."""
    if group is None:
        yield None, table, fixed or rcParams["text.color"]
        return
    if _sns_order is not None:
        levels = _sns_order(table[group])
    else:  # pragma: no cover
        levels = list(pd.unique(table[group].dropna()))
    colours = palette_for(table, group, ctx.settings)
    for level, colour in zip(levels, colours):
        yield level, table[table[group] == level], fixed or colour


def _kind_of(series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "dates"
    if pd.api.types.is_bool_dtype(series):
        return "True/False values"
    return "words (text)"


def _is_number_column(series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric_xy(word, table, x, y):
    """x and y as float arrays with empty cells skipped; friendly error if not numbers."""
    for role, col in (("x", x), ("y", y)):
        if not _is_number_column(table[col]):
            raise PlotsimpleError(f'.{word}() needs numbers on {role}, but "{col}" holds {_kind_of(table[col])}.')
    part = table[[x, y]].dropna()
    return part[x].to_numpy(dtype=float), part[y].to_numpy(dtype=float)


def _fixed_colour(word, colour):
    if colour is not None and not (isinstance(colour, str) and is_color_like(colour)):
        raise PlotsimpleError(
            f'.{word}() does not know the colour "{colour}".\n'
            'Use a colour name such as "red", "blue" or "grey", or a code such as "#0072B2".'
        )


# ====================================================================== drawing
def _not_average_error(show):
    return PlotsimpleError(
        f'.error_bars() shows spread around the average, but your lines show the {show}.\n'
        'Use chart.lines(..., show="average") with error bars.'
    )


def _draw_error_bars(ctx, kind, column, dodge=True):
    last = _layer(ctx, "error_bars", {"dots", "lines"})
    table, x, y, group = last["table"], last["x"], last["y"], last["colour"]
    ax = ctx.ax
    style = dict(capsize=4, elinewidth=1.5, capthick=1.5, zorder=3)
    bars = []  # one entry per colour group: level, colour, x values, centres, half-widths

    if column is not None:
        if is_listlike(column):
            if len(column) != len(table):
                raise PlotsimpleError(
                    f"error_bars(column=...) got {len(column)} error sizes, but the chart has {len(table)} points."
                )
            values, name = pd.Series(list(column), index=table.index), "your list"
        else:
            if column not in table.columns:
                raise column_error(column, table.columns)
            values, name = table[column], f'"{column}"'
        if not _is_number_column(values):
            raise PlotsimpleError(f".error_bars() needs numbers in {name} for the size of each error bar.")
        if (values < 0).any():
            raise PlotsimpleError(f".error_bars(): error sizes cannot be below 0, but {name} has negative values.")
        keys = [x] + ([group] if group else [])
        if last["kind"] == "lines" and table.duplicated(subset=keys).any():
            raise PlotsimpleError(
                "Your lines join several rows per x value, so one error size per row does not fit.\n"
                'Use chart.error_bars("sd") to work out the spread instead, or use one row per x value.'
            )
        data = table.assign(_err=values.to_numpy())
        for level, part, colour in _groups(ctx, data, group):
            part = part.dropna(subset=[y, "_err"])
            bars.append(dict(level=level, colour=colour, x=part[x].to_list(),
                             mid=part[y].to_numpy(dtype=float), half=part["_err"].to_numpy(dtype=float)))
        fmt = "none"
    else:
        if last.get("show", "average") not in ("average", "mean"):
            raise _not_average_error(last["show"])
        if not _is_number_column(table[y]):
            raise PlotsimpleError(f'.error_bars() needs numbers on y, but "{y}" holds {_kind_of(table[y])}.')
        summary = group_errors(table, x, y, group, kind)
        if summary["half"].isna().all():
            raise PlotsimpleError(
                f'.error_bars("{kind}") needs at least 2 values for an x value, but every x has only one.\n'
                'If your table has its own error sizes, use chart.error_bars(column="your error column").'
            )
        for level, part, colour in _groups(ctx, summary, group):
            part = part.dropna(subset=["half"])
            bars.append(dict(level=level, colour=colour, x=part[x].to_list(),
                             mid=part["mean"].to_numpy(dtype=float), half=part["half"].to_numpy(dtype=float)))
        fmt = "_" if last["kind"] == "dots" else "none"  # on dots, also mark the average

    positions = [list(b["x"]) for b in bars]
    if dodge and group is not None:
        shifted = _dodge(ax, last, bars)
        if shifted is not None:
            positions = shifted
    for b, xs in zip(bars, positions):
        if len(b["mid"]):
            ax.errorbar(xs, b["mid"], yerr=b["half"], fmt=fmt, color=b["colour"], ecolor=b["colour"],
                        markersize=16, markeredgewidth=2, **style)


# ---------------------------------------------------------------- dodging grouped error bars
DODGE_SPREAD = 0.15  # the groups spread over 15 % of the smallest gap between x values


def _to_numbers(ax, values) -> np.ndarray:
    """x values as the numbers matplotlib uses on the axis (categories 0, 1, 2 ...; dates in days)."""
    if len(values) == 0:
        return np.array([], dtype=float)
    return np.asarray(ax.xaxis.convert_units(list(values)), dtype=float)


def dodge_offsets(n_groups: int, gap: float) -> np.ndarray:
    """Sideways shift for each group, in legend order, centred on the true x (ggplot2 position_dodge)."""
    if n_groups < 2:
        return np.zeros(n_groups)
    spread = DODGE_SPREAD * gap
    return (np.arange(n_groups) - (n_groups - 1) / 2) * spread / (n_groups - 1)


def _same_rgb(a, b) -> bool:
    return np.allclose(to_rgba(a)[:3], to_rgba(b)[:3], atol=1e-3)


def _same_x(a, b) -> bool:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    return len(a) == len(b) and np.allclose(a, b)


def _dodge(ax, last, bars):
    """Shift each group's error bars, line and dots sideways by the same amount.

    Returns the new x positions per group, or None when nothing should move: no x value
    is shared by 2 groups, or a group's own line/dots cannot be found for certain (then the
    bars stay on the true x rather than come loose from their points).
    """
    table, x, y, group = last["table"], last["x"], last["y"], last["colour"]
    numbers = [_to_numbers(ax, b["x"]) for b in bars]
    owners = {}
    for i, xs in enumerate(numbers):
        for v in np.unique(xs):
            owners.setdefault(v, set()).add(i)
    if not any(len(o) > 1 for o in owners.values()):
        return None
    if _is_number_column(table[x]) or pd.api.types.is_datetime64_any_dtype(table[x]):
        distinct = np.unique(np.concatenate(numbers))
        if len(distinct) < 2:
            return None
        gap = float(np.diff(distinct).min())
    else:  # categories sit 1 apart
        gap = 1.0
    offsets = dodge_offsets(len(bars), gap)

    # find every group's own artists first; move nothing unless all are found
    moves = []
    points = table.dropna(subset=[x, y])
    for b in bars:
        part = points[points[group] == b["level"]]
        own_x = _to_numbers(ax, part[x].to_list())
        if last["kind"] == "lines":
            found = [ln for ln in ax.lines
                     if len(ln.get_xdata()) and _same_rgb(ln.get_color(), b["colour"])
                     and _same_x(ln.convert_xunits(ln.get_xdata()), np.unique(own_x))]
            if len(found) != 1:
                return None
            moves.append(("line", found[0]))
        else:
            hits = []
            for coll in ax.collections:
                if not isinstance(coll, PathCollection):
                    continue
                offs = np.asarray(coll.get_offsets())
                faces = coll.get_facecolors()
                if len(offs) == 0 or len(faces) != len(offs):
                    continue
                mask = np.array([_same_rgb(f, b["colour"]) for f in faces])
                if mask.any() and _same_x(offs[mask, 0], own_x):
                    hits.append((coll, mask))
            if len(hits) != 1:
                return None
            moves.append(("dots", hits[0]))
    for (what, target), off in zip(moves, offsets):
        if what == "line":
            target.set_xdata(np.asarray(target.convert_xunits(target.get_xdata()), dtype=float) + off)
        else:
            coll, mask = target
            offs = np.array(coll.get_offsets(), dtype=float)
            offs[mask, 0] += off
            coll.set_offsets(offs)
    return [xs + off for xs, off in zip(numbers, offsets)]


def _draw_fit(ctx, word, how, colour, show_equation, band=False, degree=1):
    last = _layer(ctx, word, {"dots"})
    table, x, y, group = last["table"], last["x"], last["y"], last["colour"]
    _numeric_xy(word, table, x, y)
    ax = ctx.ax
    drawn = 0
    need = {"line": 2, "curve": degree + 1, "smooth": 3}[how]
    for name, part, col in _groups(ctx, table, group, colour):
        xv, yv = _numeric_xy(word, part, x, y)
        if len(np.unique(xv)) < need:
            continue
        xs = np.linspace(xv.min(), xv.max(), 200)  # only over the x range of the data
        if how == "line":
            fit = linear_fit(xv, yv)
            ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=col, zorder=3)
            if band and "s" in fit:
                low, high = line_band(fit, xs)
                ax.fill_between(xs, low, high, color=col, alpha=0.2, linewidth=0, zorder=2.5)
            coefs, r2 = [fit["slope"], fit["intercept"]], fit["r2"]
        elif how == "curve":
            fit = poly_fit(xv, yv, degree)
            ax.plot(xs, np.polyval(fit["coefs"], xs), color=col, zorder=3)
            coefs, r2 = fit["coefs"], fit["r2"]
        else:
            delta = 0.01 * np.ptp(xv) if len(xv) > 1000 else 0.0
            xs, ys = lowess(xv, yv, delta=delta)
            ax.plot(xs, ys, color=col, zorder=3)
        drawn += 1
        if show_equation and how != "smooth":
            text = equation_text(coefs, r2)
            _write_equation(ctx, text if name is None else f"{name}: {text}", col)
    if not drawn:
        raise PlotsimpleError(
            f".{word}() needs dots at {need} or more different x values"
            + (" in at least one colour group." if group else ".")
        )


def _draw_fit_line(ctx, band, show_equation, colour):
    _draw_fit(ctx, "fit_line", "line", colour, show_equation, band=band)


def _draw_fit_curve(ctx, degree, show_equation, colour):
    _draw_fit(ctx, "fit_curve", "curve", colour, show_equation, degree=degree)


def _draw_smooth_line(ctx, colour):
    _draw_fit(ctx, "smooth_line", "smooth", colour, False)


def _draw_side_histograms(ctx):
    last = _layer(ctx, "side_histograms", {"dots"})
    table, x, y, group = last["table"], last["x"], last["y"], last["colour"]
    _numeric_xy("side_histograms", table, x, y)
    ax = ctx.ax
    top = ax.inset_axes([0, 1.02, 1, 0.18], sharex=ax)
    right = ax.inset_axes([1.02, 0, 0.18, 1], sharey=ax)
    edges = {}
    for col in (x, y):
        values = table[col].dropna().to_numpy(dtype=float)
        e = np.histogram_bin_edges(values, bins="auto")
        edges[col] = e if len(e) <= 51 else np.histogram_bin_edges(values, bins=50)
    many = group is not None
    fixed = None if many else default_colour(ctx.settings)
    for _, part, colour in _groups(ctx, table, group, fixed):
        if many:  # outlines, so overlapping groups stay readable
            kw = dict(color=colour, histtype="step", linewidth=1.5)
        else:
            kw = dict(color=colour, alpha=0.8, histtype="stepfilled", linewidth=0)
        top.hist(part[x].dropna().to_numpy(dtype=float), bins=edges[x], **kw)
        right.hist(part[y].dropna().to_numpy(dtype=float), bins=edges[y], orientation="horizontal", **kw)
    for side in (top, right):
        side.grid(False)
        for spine in side.spines.values():
            spine.set_visible(False)
    top.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    right.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)


def _draw_line_at(ctx, axis, value, dashed, colour, label):
    ax = ctx.ax
    style = dict(color=colour, linestyle="--" if dashed else "-", linewidth=1.5, zorder=2.5)
    size = rcParams["legend.fontsize"]
    back = dict(boxstyle="round,pad=0.15", fc=rcParams["axes.facecolor"], ec="none", alpha=0.85)
    if axis == "x":
        ax.axvline(value, **style)
        if label:
            ax.text(value, 0.98, f" {label} ", transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", color=colour, fontsize=size, bbox=back)
    else:
        ax.axhline(value, **style)
        if label:
            # just outside the right edge, so it never sits on top of bars or dots
            ax.text(1.01, value, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", color=colour, fontsize=size, clip_on=False)


def _draw_shade(ctx, axis, low, high, colour):
    span = ctx.ax.axvspan if axis == "x" else ctx.ax.axhspan
    span(low, high, color=colour, alpha=0.2, linewidth=0, zorder=0)


def _check_position(word, option, value, example):
    ok = not isinstance(value, bool) and isinstance(
        value, (int, float, np.integer, np.floating, _dt.date, np.datetime64))
    if ok and isinstance(value, (float, np.floating)) and not np.isfinite(value):
        ok = False
    if not ok:
        raise PlotsimpleError(f".{word}() needs a number for {option}=, not {value!r}.\nFor example chart.{example}")


# ====================================================================== words
class StatsWords:
    def _check_layer_now(self, word, kinds, numeric=True):
        """Report a wrong layer on the user's own line when it can already be seen.

        If no chart type has been added yet that is fine: steps may come in any order,
        and the check is made again when the chart is drawn.
        """
        draws = [s for s in self._steps if s.phase == PHASES["draw"] and s.name in _CHART_KINDS]
        if not draws:
            return None
        step = draws[-1]
        if step.name not in kinds:
            raise _wrong_layer(word, kinds, step.name)
        if numeric and len(self._data.columns):
            for role in ("x", "y"):
                col = step.kwargs.get(role)
                if isinstance(col, str) and col in self._data.columns and not _is_number_column(self._data[col]):
                    raise PlotsimpleError(
                        f'.{word}() needs numbers on {role}, but "{col}" holds {_kind_of(self._data[col])}.')
        return step

    @friendly
    def error_bars(self, kind="sd", column=None, dodge=True):
        """Error bars on dots or lines. chart.error_bars("sd")

        For every x value the bar shows the average of y and its spread:
        "sd" standard deviation (n - 1), "se" standard error, "ci" 95 % confidence interval
        (t distribution). Empty cells are skipped.
        column="error" uses error sizes from your own column instead.
        With colour groups, bars at the same x move a little apart (with their lines and
        dots) so they do not hide each other; dodge=False keeps them on the true x.
        """
        self._check_choice("error_bars", "kind", kind, ERRORS)
        if column is not None and not is_listlike(column):
            if not isinstance(column, str):
                raise PlotsimpleError('column= should be the name of a column, for example column="error".')
            if len(self._data.columns) and column not in self._data.columns:
                raise column_error(column, self._data.columns)
        step = self._check_layer_now("error_bars", {"dots", "lines"}, numeric=False)
        if step is not None and step.name == "lines" and column is None:
            show = step.kwargs.get("show", "average")
            if show not in ("average", "mean"):
                raise _not_average_error(show)
        return self._add("error_bars", "extra", _draw_error_bars, kind=kind, column=column,
                         dodge=bool(dodge))

    @friendly
    def fit_line(self, band=False, show_equation=False, colour=None):
        """A straight line that fits the dots best. chart.fit_line()

        band=True shades the 95 % confidence band of the line.
        show_equation=True writes the equation and R² on the chart.
        Each colour group gets its own line. colour="red" makes every line red.
        """
        _fixed_colour("fit_line", colour)
        self._check_layer_now("fit_line", {"dots"})
        return self._add("fit_line", "extra", _draw_fit_line, band=bool(band),
                         show_equation=bool(show_equation), colour=colour)

    @friendly
    def fit_curve(self, degree=2, show_equation=False, colour=None):
        """A curve that fits the dots best. chart.fit_curve(degree=2)

        degree=2 is a curve with one bend (x²), degree=3 can bend twice (x³), and so on.
        show_equation=True writes the equation and R² on the chart.
        """
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 10:
            raise PlotsimpleError(
                f"degree= should be a whole number from 1 to 10, not {degree!r}.\n"
                "For example chart.fit_curve(degree=2) for a curve with one bend."
            )
        _fixed_colour("fit_curve", colour)
        self._check_layer_now("fit_curve", {"dots"})
        return self._add("fit_curve", "extra", _draw_fit_curve, degree=int(degree),
                         show_equation=bool(show_equation), colour=colour)

    @friendly
    def smooth_line(self, colour=None):
        """A smooth line that follows the dots (LOWESS). chart.smooth_line()

        Good for seeing a trend that is not straight. Each colour group gets its own line.
        """
        _fixed_colour("smooth_line", colour)
        self._check_layer_now("smooth_line", {"dots"})
        return self._add("smooth_line", "extra", _draw_smooth_line, colour=colour)

    @friendly
    def side_histograms(self):
        """Small histograms along the top and right edges of a dots chart. chart.side_histograms()"""
        self._check_layer_now("side_histograms", {"dots"})
        return self._add("side_histograms", "extra", _draw_side_histograms)

    @friendly
    def line_at_x(self, value, dashed=True, colour="grey", label=None):
        """A straight up-and-down line at one x value. chart.line_at_x(2010, label="2010")"""
        _check_position("line_at_x", "value", value, "line_at_x(10)")
        _fixed_colour("line_at_x", colour)
        return self._add("line_at_x", "extra", _draw_line_at, axis="x", value=value, dashed=bool(dashed),
                         colour=colour, label=None if label is None else str(label))

    @friendly
    def line_at_y(self, value, dashed=True, colour="grey", label=None):
        """A straight side-to-side line at one y value. chart.line_at_y(50, label="pass mark")"""
        _check_position("line_at_y", "value", value, "line_at_y(50)")
        _fixed_colour("line_at_y", colour)
        return self._add("line_at_y", "extra", _draw_line_at, axis="y", value=value, dashed=bool(dashed),
                         colour=colour, label=None if label is None else str(label))

    def _shade(self, axis, low, high, colour):
        word = f"shade_between_{axis}"
        example = f"{word}(40, 60)"
        _check_position(word, "low", low, example)
        _check_position(word, "high", high, example)
        _fixed_colour(word, colour)
        if low == high:
            raise PlotsimpleError(f".{word}() needs two different values, for example chart.{example}")
        if high < low:
            low, high = high, low
        return self._add(word, "extra", _draw_shade, axis=axis, low=low, high=high, colour=colour)

    @friendly
    def shade_between_x(self, low, high, colour="grey"):
        """Shade the band between two x values, behind the data. chart.shade_between_x(6, 8)"""
        return self._shade("x", low, high, colour)

    @friendly
    def shade_between_y(self, low, high, colour="grey"):
        """Shade the band between two y values, behind the data. chart.shade_between_y(40, 60)"""
        return self._shade("y", low, high, colour)


# ====================================================================== ps.pairs
class PairsChart(TextWords, LookWords, BaseChart):
    """Every number column against every other. Make one with ps.pairs(table)."""

    def __init__(self, data, colour=None, columns=None):
        super().__init__(data)
        table = self._data
        if not len(table.columns):
            raise PlotsimpleError('ps.pairs() needs a table, for example ps.pairs("measurements.xlsx").')
        group, fixed = colour_role(table, colour)
        if is_listlike(group):
            raise PlotsimpleError('In ps.pairs(), colour= should be a column name, for example colour="species".')
        if columns is not None:
            if isinstance(columns, str) or not is_listlike(columns):
                raise PlotsimpleError('columns= should be a list of column names, for example columns=["a", "b"].')
            for c in columns:
                if c not in table.columns:
                    raise column_error(c, table.columns)
            chosen = list(columns)
        else:
            chosen = [c for c in table.columns if c != group]
        numbers = [c for c in chosen if _is_number_column(table[c])]
        if columns is not None and len(numbers) < len(chosen):
            bad = [str(c) for c in chosen if c not in numbers]
            raise PlotsimpleError("ps.pairs() can only use columns with numbers. Not numbers: " + ", ".join(bad))
        if len(numbers) < 2:
            raise PlotsimpleError(
                "ps.pairs() needs at least 2 columns with numbers.\n"
                f"Columns with numbers: {', '.join(map(str, numbers)) or 'none'}"
            )
        if len(numbers) > 10:
            raise PlotsimpleError(
                f"Your table has {len(numbers)} columns with numbers, too many for one pairs chart.\n"
                'Choose up to 10 with columns=, for example ps.pairs(table, columns=["a", "b", "c"]).'
            )
        self._columns = numbers
        self._group, self._fixed = group, fixed
        side = float(min(max(2.4 * len(numbers), 5), 12))
        self._settings.update(width=side + (1.5 if group else 0), height=side)

    def label_x(self, *args, **kwargs):
        raise PlotsimpleError("A pairs chart already names each column under its panels.")

    def label_y(self, *args, **kwargs):
        raise PlotsimpleError("A pairs chart already names each row beside its panels.")

    def __repr__(self):
        return f"<plotsimple pairs chart: {', '.join(map(str, self._columns))}>"

    def _render(self) -> Figure:
        if self._cache is not None:
            return self._cache
        s = self._settings
        table, cols, group = self._data, self._columns, self._group
        k = len(cols)
        with _style.context(s):
            fig = Figure(figsize=(s["width"], s["height"]), layout="constrained")
            axes = fig.subplots(k, k, sharex="col", sharey="row", squeeze=False)
            colour_kw = (dict(hue=group, palette=palette_for(table, group, s)) if group
                         else dict(color=default_colour(s, self._fixed)))
            handles = None
            dot_size = max(8, 40 - 4 * k)
            for i, row_col in enumerate(cols):
                for j, col in enumerate(cols):
                    ax = axes[i][j]
                    if i == j:
                        # the cell keeps the row's y scale (shared); the histogram sits on a twin
                        twin = ax.twinx()
                        if group:
                            sns.kdeplot(data=table, x=col, ax=twin, fill=True, common_norm=False,
                                        warn_singular=False, legend=False, **colour_kw)
                        else:
                            sns.histplot(data=table, x=col, ax=twin, **colour_kw)
                        twin.set_ylim(bottom=0)
                        twin.set_ylabel("")
                        twin.grid(False)
                        twin.tick_params(right=False, labelright=False)
                        twin.spines["right"].set_visible(False)
                    else:
                        want_legend = group is not None and handles is None
                        sns.scatterplot(data=table, x=col, y=row_col, ax=ax, s=dot_size, linewidth=0,
                                        alpha=0.8, legend="auto" if want_legend else False, **colour_kw)
                        if want_legend and ax.get_legend() is not None:
                            handles = ax.get_legend_handles_labels()
                            ax.get_legend().remove()
                    ax.set_xlabel(str(col) if i == k - 1 else "")
                    ax.set_ylabel(str(row_col) if j == 0 else "")
            if handles:
                fig.legend(*handles, title=str(group), loc="outside right upper")
            self._draw_figure_steps(fig, list(axes.flat))
        self._cache = fig
        return fig


def pairs(data, colour=None, columns=None):
    """Every number column against every other. ps.pairs(ps.example("marks"), colour="class")

    Histograms on the diagonal (density curves when colour= makes groups), dots
    everywhere else. colour="class" colours each group. columns=["hours", "marks"] chooses
    the columns. Empty cells are skipped. Then use it like any chart:
    .title_top("..."), .save("pairs.png").
    """
    return PairsChart(data, colour=colour, columns=columns)
