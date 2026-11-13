"""Curves on charts: a formula, a fitted formula, a fitted distribution, and lines joined by curves.

* curve(): any formula of one name (x), worked out with the shared safe formula reader
  (_maths_formula.read_formula, never run), at steps= evenly spaced places over between=
  (or the x range of the chart). Or a distribution: its density curve (continuous) or the
  chance of each whole number (discrete); show="cumulative" draws the chance of a value at
  or below x instead. Over a histogram the density is scaled to what its bars show:
  count = density x number of values x bar width, percent = density x 100 x bar width,
  density as it is (a note says so).
* fit_formula(): table.fit_formula's engine (table/fitting.fit_formula_core: nonlinear least
  squares, as R nls()) on the dots of each colour group; the curve is the formula with the
  values found, over each group's x range.
* fit_distribution(): the maximum-likelihood fit of the distributions engine (as R
  MASS::fitdistr()) on the values of a histogram or density chart, per colour group, scaled
  as curve() scales a density.
* lines(joined_by=) and qq_dots(distribution=) use the helpers at the end of this file;
  the joining itself is table/between.py (the engine of table.values_at()).
"""

from __future__ import annotations

import contextlib
import io
import math
import re

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import is_color_like

from .._core import chart_note
from .._errors import SimplydataError, choice_error, friendly
from ._helpers import check_thickness, palette_for

CURVE_SHOW = ("density", "cumulative")
CURVE_STEPS_MOST = 100_000


# ====================================================================== shared helpers
def _is_number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(v)


def _num(x) -> str:
    from ..table.stat_tests import _num as num

    return num(x)


def _check_between(word, between, example):
    if between is None:
        return None
    if isinstance(between, str) or not hasattr(between, "__len__") or len(between) != 2 \
            or not all(_is_number(v) for v in between) or not between[0] < between[1]:
        raise SimplydataError(f".{word}() needs between= as [low, high] with low below high, such as "
                              f"between=[0, 10], not {between!r}.\nFor example chart.{example}")
    return (float(between[0]), float(between[1]))


def _check_colour(word, colour):
    if colour is not None and not (isinstance(colour, str) and is_color_like(colour)):
        raise SimplydataError(f'.{word}() does not know the colour "{colour}".\n'
                              'Use a colour name such as "red", "blue" or "grey", or a code such as "#0072B2".')


def _x_range_of_chart(ctx):
    """The x range of what is drawn so far, or None for an empty chart."""
    ax = ctx.ax
    if ctx.last is None and not ax.lines and not ax.collections and not ax.patches:
        return None
    low, high = ax.dataLim.intervalx
    if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
        return None
    return float(low), float(high)


def _say_once(said, key, line):
    """Print a line once for this step (a chart may be drawn many times), at the end of the picture."""
    if said.get(key):
        return
    said[key] = True
    chart_note(("curves", id(said), key), lambda _count, line=line: line)


def _legend_entry(ctx, line, label):
    if label:
        ctx.store.setdefault("slope_lines", []).append((line, label))


# ---------------------------------------------------------------------- formulas
def read_curve_formula(word, text):
    """(function of one name, the y name or None, the name, the formula text) from "y = 100 * exp(-0.3 * x)"."""
    from .._maths_formula import read_formula

    example = 'curve("y = 100 * exp(-0.3 * x)", between=[0, 10])'
    if not isinstance(text, str) or not text.strip():
        raise SimplydataError(f".{word}() needs a formula in quotes or a distribution.\nFor example chart.{example}")
    left = None
    right = text.strip()
    if right.count("=") == 1 and not re.search(r"[<>!]=|==", right):
        left, right = (part.strip() for part in right.split("="))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", left or ""):
            raise SimplydataError(f'.{word}() reads "{text}" as "name = formula"; the left of = should be a name, '
                                  f"such as y.\nFor example chart.{example}")
    formula = read_formula(right)
    if len(formula.names) != 1:
        if not formula.names:
            raise SimplydataError(f'"{right}" has no name such as x in it, so it does not change along the chart. '
                                  f"For a flat line use chart.line_at_y(...).\nFor example chart.{example}")
        others = ", ".join(formula.names[1:])
        raise SimplydataError(
            f'.{word}() draws a formula of one name, along x; "{right}" has {len(formula.names)}: '
            f"{', '.join(formula.names)}. Put numbers in place of {others}.\nFor example chart.{example}")
    return formula, left, formula.names[0], right


# ---------------------------------------------------------------------- distributions
def _is_distribution(thing) -> bool:
    return hasattr(thing, "is_discrete") and any(hasattr(thing, n) for n in ("_pdf", "_pmf", "pdf", "pmf"))


def _density(dist, x):
    """Height of the density curve: a simplydata Distribution keeps pdf private (dist.pdf() points
    people to density_at()); other objects with .pdf work too."""
    return np.asarray((getattr(dist, "_pdf", None) or dist.pdf)(x), dtype=float)


def _chance_of(dist, k):
    return np.asarray((getattr(dist, "_pmf", None) or dist.pmf)(k), dtype=float)


def _frozen(dist):
    """The SciPy frozen distribution inside a Distribution, if it shows one."""
    for name in ("frozen", "_frozen", "scipy", "_scipy", "_dist", "dist"):
        inner = getattr(dist, name, None)
        if inner is not None and hasattr(inner, "ppf") and not callable(getattr(inner, "is_discrete", None)):
            return inner
    return None


def quantiles(dist, shares):
    """The values with these shares (0 to 1) of the distribution below them (the ppf), as an array."""
    shares = np.asarray(shares, dtype=float)
    inner = _frozen(dist)
    if inner is not None:
        return np.asarray(inner.ppf(shares), dtype=float)
    if hasattr(dist, "ppf"):
        return np.asarray(dist.ppf(shares), dtype=float)
    out = dist.percentile(list(shares * 100))
    return np.asarray(list(out) if not np.isscalar(out) else [out], dtype=float)


def _cumulative(dist, x):
    inner = _frozen(dist)
    if inner is not None:
        return np.asarray(inner.cdf(x), dtype=float)
    if hasattr(dist, "_cdf") or hasattr(dist, "cdf"):
        return np.asarray((getattr(dist, "_cdf", None) or dist.cdf)(x), dtype=float)
    out = dist.chance(at_most=list(np.asarray(x, dtype=float)))
    return np.asarray(list(out), dtype=float)


def _label_of(dist) -> str:
    label = getattr(dist, "label", None)
    return str(label() if callable(label) else label) if label is not None else "the distribution"


def _default_range(dist):
    """From the 0.1 % to the 99.9 % point (the support's ends where it is shorter)."""
    low, high = quantiles(dist, [0.001, 0.999])
    return float(low), float(high)


def fit_distribution_to(values, name, column=None, say=False):
    """The maximum-likelihood fit of the distributions engine, and the sentences it printed
    (the same as table.fit_distribution's when say=True), kept to be printed once."""
    from ..distributions import fit

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        dist = fit(values, name, say=say, column=column)
    return dist, out.getvalue().strip()


# ---------------------------------------------------------------------- histogram scaling
def histogram_scale(ctx, n_values, cumulative=False):
    """(factor, words) to put a density (or cumulative chance) on the last histogram's scale."""
    last = ctx.last or {}
    if last.get("kind") != "histogram":
        return 1.0, ""
    shown = ctx.ax.get_ylabel().strip()
    is_cumulative = shown.startswith("cumulative")
    show = shown.replace("cumulative", "").strip() or "count"
    if cumulative or is_cumulative:
        if not (cumulative and is_cumulative):
            raise SimplydataError(
                'On a cumulative histogram draw the cumulative curve: chart.curve(distribution, show="cumulative"); '
                'on an ordinary histogram show="density".' if is_cumulative else
                'show="cumulative" goes on a cumulative histogram: chart.histogram(..., cumulative=True).')
        factor = {"count": float(n_values), "percent": 100.0, "density": 1.0}.get(show, 1.0)
        words = {"count": f"x {n_values} values", "percent": "x 100", "density": ""}.get(show, "")
        return factor, (f"to the histogram's cumulative {show} ({words.strip()})" if words else "")
    widths = [p.get_width() for p in ctx.ax.patches if hasattr(p, "get_width") and p.get_width() > 0]
    if not widths:
        return 1.0, ""
    width = float(np.median(widths))
    if show == "density":
        return 1.0, ""
    if not np.allclose(widths, width, rtol=1e-6):
        chart_note(("curves", "uneven"), lambda _c: "Note: the histogram's bars are not all the same width, so the "
                   "curve is scaled with the usual (median) width; use show=\"density\" for an exact match.")
    if show == "percent":
        return 100.0 * width, f"to the histogram's percent (density x 100 x bar width {_num(width)})"
    return n_values * width, f"to the histogram's counts (density x {n_values} values x bar width {_num(width)})"


def _values_of_layer(ctx, word):
    last = ctx.last or {}
    table, column = last.get("table"), last.get("x")
    if table is None or column is None:
        return None
    return pd.to_numeric(table[column], errors="coerce")


_HISTOGRAM_EXAMPLE = 'chart.histogram(x="marks", show="density")'


def _histogram_layer_error(found):
    if found is None:
        return SimplydataError(".fit_distribution() works on histogram or density, but your chart has none yet.\n"
                               f"Add one first, for example {_HISTOGRAM_EXAMPLE}")
    return SimplydataError(f".fit_distribution() works on histogram or density, but the last chart type is {found}.\n"
                           f"Use it after {_HISTOGRAM_EXAMPLE}")


# ====================================================================== drawing
def _draw_curve(ctx, thing, between, steps, dashed, colour, label, thickness, show, said):
    ax = ctx.ax
    colour = colour or rcParams["text.color"]
    style = dict(color=colour, linestyle="--" if dashed else "-", zorder=3,
                 linewidth=2 if thickness is None else thickness)
    if isinstance(thing, tuple):  # a formula
        formula, left, name, text = thing
        span = between or _x_range_of_chart(ctx)
        if span is None:
            raise SimplydataError(f'.curve("{text}") needs between=[low, high] to know where to draw, because the '
                                  "chart has nothing else on it yet.\nFor example chart.curve(\"y = 100 * exp(-0.3 * x)\", "
                                  "between=[0, 10])")
        xs = np.linspace(span[0], span[1], steps)
        ys = np.asarray(formula({name: xs}), dtype=float) * np.ones_like(xs)
        line, = ax.plot(xs, ys, **style)
        _legend_entry(ctx, line, label)
        if ctx.last is None:
            if not ax.get_xlabel():
                ax.set_xlabel(name)
            if not ax.get_ylabel():
                ax.set_ylabel(left or text)
        return
    dist = thing
    cumulative = show == "cumulative"
    span = between or (_x_range_of_chart(ctx) if ctx.last is not None else None) or _default_range(dist)
    values = _values_of_layer(ctx, "curve")
    n_values = int(values.notna().sum()) if values is not None else 0
    factor, words = histogram_scale(ctx, n_values, cumulative)
    if words:
        _say_once(said, "scale", f"Note: the curve of {_label_of(dist)} was scaled {words}.")
    if dist.is_discrete:
        ks = np.arange(math.ceil(span[0]), math.floor(span[1]) + 1)
        heights = (_cumulative(dist, ks) if cumulative else _chance_of(dist, ks)) * factor
        if ctx.last is None and not cumulative:
            bars = ax.bar(ks, heights, width=0.8, color=colour, alpha=0.7, zorder=2.5)
            handle = bars
        elif cumulative:
            handle, = ax.plot(ks, heights, drawstyle="steps-post", **style)
        else:
            ax.vlines(ks, 0, heights, color=colour, linewidth=style["linewidth"], zorder=3)
            handle, = ax.plot(ks, heights, "o", color=colour, zorder=3.1)
        _legend_entry(ctx, handle, label)
        if ctx.last is None:
            ax.set_ylabel("chance at or below" if cumulative else "chance")
        return
    xs = np.linspace(span[0], span[1], steps)
    ys = (_cumulative(dist, xs) if cumulative else _density(dist, xs)) * factor
    line, = ax.plot(xs, ys, **style)
    _legend_entry(ctx, line, label)
    if ctx.last is None:
        if not ax.get_ylabel():
            ax.set_ylabel("chance at or below" if cumulative else "density")
        if not ax.get_xlabel():
            ax.set_xlabel("value")


def _formula_with_values(text, values: dict) -> str:
    for name, value in values.items():
        text = re.sub(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_(])", _num(value), text)
    return text


def _colour_groups(ctx, table, group, fixed):
    from .stats import _groups

    yield from _groups(ctx, table, group, fixed)


def _draw_fit_formula(ctx, formula, start, between, show_equation, colour, thickness, said):
    from ..table import fitting
    from .stats import _layer, _numeric_xy, _write_equation

    last = _layer(ctx, "fit_formula", {"dots"})
    table, x, y, group = last["table"], last["x"], last["y"], last["colour"]
    _numeric_xy("fit_formula", table, x, y)
    parsed = fitting.read_fit_formula(formula, list(table.columns), "fit_formula")
    if parsed.left != y:
        raise SimplydataError(f'.fit_formula() fits the y of the dots ("{y}"), so the formula should start with '
                              f'"{y} = ...", not "{parsed.left} = ...".')
    others = [c for c in parsed.used if c != x]
    if others:
        raise SimplydataError(f'.fit_formula() draws the curve along x ("{x}"), so the formula can use only that column; '
                              f"it also uses {', '.join(others)}. Use table.fit_formula() for formulas of several columns.")
    if x not in parsed.used:
        raise SimplydataError(f'The formula has no "{x}" in it, so it would not change along the chart.')
    lines, notes, drawn, method = [], [], 0, ""
    for name, part, col in _colour_groups(ctx, table, group, colour):
        xv, yv = _numeric_xy("fit_formula", part, x, y)
        if len(xv) <= len(parsed.names) and group is not None:
            notes.append(f'Note: .fit_formula() left out "{name}": it has too few dots for {len(parsed.names)} '
                         "values to find.")
            continue
        try:
            fit = fitting.fit_formula_core(parsed, {y: yv, x: xv}, list(table.columns), start=start,
                                           between=between, step="fit_formula")
        except SimplydataError as error:
            if group is None:
                raise
            notes.append(f'Note: .fit_formula() could not fit the group "{name}": {str(error).splitlines()[0]}')
            continue
        xs = np.linspace(xv.min(), xv.max(), 200)
        ctx.ax.plot(xs, fit.predict({x: xs}), color=col, zorder=3, linewidth=thickness)
        drawn += 1
        low, high = fit.intervals
        found = ", ".join(f"{k} = {_num(v)} (95 % CI {_num(lo)} to {_num(hi)})"
                          for k, v, lo, hi in zip(fit.names, fit.values, low, high))
        where = "" if name is None else f' for "{name}"'
        lines.append(f'Fitted "{parsed.text}"{where} to {fit.n} rows: {found}. Residual standard error '
                     f"{_num(fit.sigma)} on {fit.df} degrees of freedom.")
        notes.extend(fit.notes)
        method = fit.method_sentence
        if show_equation:
            text = f"{y} = {_formula_with_values(parsed.text.split('=', 1)[1].strip(), fit.value_dict)}"
            _write_equation(ctx, text if name is None else f"{name}: {text}", col)
    if not drawn:
        raise SimplydataError(".fit_formula() could not fit the formula to any group of dots.")
    _say_once(said, "fit", "\n".join(lines + [method] + [n for n in notes if n]))


def _draw_fit_distribution(ctx, distribution, colour, thickness, said):
    last = ctx.last
    if not last or last.get("kind") not in ("histogram", "density"):
        raise _histogram_layer_error(last.get("kind") if last else None)
    table, x, group = last["table"], last["x"], last["colour"]
    lines = []
    for name, part, col in _colour_groups(ctx, table, group, colour):
        values = pd.to_numeric(part[x], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        # one curve: the sentences of table.fit_distribution; groups: one line each
        dist, printed = fit_distribution_to(values, distribution, column=x, say=name is None)
        span = _x_range_of_chart(ctx) or _default_range(dist)
        factor, words = histogram_scale(ctx, len(values))
        width = 2 if thickness is None else thickness
        if dist.is_discrete:  # counts: the chance of each whole number, as dots on thin lines
            ks = np.arange(math.ceil(span[0]), math.floor(span[1]) + 1)
            heights = _chance_of(dist, ks) * factor
            ctx.ax.vlines(ks, 0, heights, color=col, linewidth=width, zorder=3)
            ctx.ax.plot(ks, heights, "o", color=col, zorder=3.1)
        else:
            xs = np.linspace(span[0], span[1], 200)
            ctx.ax.plot(xs, _density(dist, xs) * factor, color=col, zorder=3, linewidth=width)
        where = f'"{x}"' if name is None else f'"{x}" in "{name}"'
        lines.append(printed or f"Fitted {_label_of(dist)} to {where} ({len(values)} values), by maximum "
                                 "likelihood (as R MASS::fitdistr()).")
        if words:
            lines.append(f"Note: the fitted curve was scaled {words}.")
    if not lines:
        raise SimplydataError(".fit_distribution() needs at least 2 values to fit.")
    _say_once(said, "fit", "\n".join(dict.fromkeys(lines)))


# ====================================================================== words
class CurveWords:
    @friendly
    def curve(self, formula_or_distribution=None, between=None, steps=200, dashed=False, colour=None, label=None,
              thickness=None, show="density"):
        """Draw a formula, or a distribution, as a curve. chart.curve("y = 100 * exp(-0.3 * x)", between=[0, 10])

        The formula uses one name (x, or any other) and the functions of table.add_column;
        between=[low, high] says where along x (normally the x range of the chart).
        chart.curve(sdata.distribution("normal", average=50, standard_deviation=10)) draws a
        distribution: its density curve, or for whole-number distributions (binomial,
        poisson ...) the chance of each value. Over a histogram the curve is scaled to the
        bars (count, percent or density), with a note. show="cumulative" draws the chance of
        a value at or below x instead (on a cumulative histogram it is scaled the same way).
        steps= is how many points make the curve (normally 200); dashed=True, colour="red",
        thickness=3 and label="model" (shown in the legend) change how it looks.
        """
        word = "curve"
        thickness = check_thickness(word, thickness)
        _check_colour(word, colour)
        span = _check_between(word, between, 'curve("y = 100 * exp(-0.3 * x)", between=[0, 10])')
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or not 2 <= steps <= CURVE_STEPS_MOST:
            raise SimplydataError(f"steps= should be a whole number from 2 to {CURVE_STEPS_MOST:,}, such as steps=200, "
                                  f"not {steps!r}.")
        if show not in CURVE_SHOW:
            raise choice_error(word, "show", show, CURVE_SHOW)
        if _is_distribution(formula_or_distribution):
            thing = formula_or_distribution
        else:
            if show == "cumulative":
                raise SimplydataError('show="cumulative" is for a distribution, such as '
                                      'chart.curve(sdata.distribution("normal"), show="cumulative").')
            thing = read_curve_formula(word, formula_or_distribution)
        self._add(word, "extra", _draw_curve, thing=thing, between=span, steps=int(steps), dashed=bool(dashed),
                  colour=colour, label=None if label is None else str(label), thickness=thickness, show=show,
                  said={})
        if label is not None:
            from .stats import _slope_legend

            self._replace("slope_legend", "extra", _slope_legend)
        return self

    @friendly
    def fit_formula(self, formula=None, start=None, between=None, show_equation=False, colour=None, thickness=None):
        """The curve of a formula fitted to the dots. chart.fit_formula("population = a * exp(b * (year - 2000))")

        After dots, for example chart.dots(x="year", y="population", colour="country") (with
        sdata.example("growth")).

        The same fit as table.fit_formula (nonlinear least squares, as R nls()): names that
        are columns are data, other names (a, b) are the values to find. start= and
        between= work as there. Each colour group gets its own curve, and the values found
        are printed (once). show_equation=True writes the formula with the values found on
        the chart. colour="red" makes every curve red; thickness=3 draws thicker curves.
        """
        word = "fit_formula"
        thickness = check_thickness(word, thickness)
        _check_colour(word, colour)
        if not isinstance(formula, str) or formula.count("=") != 1:
            raise SimplydataError(
                '.fit_formula() needs a formula with one = sign: the y column on the left, for example\n'
                'chart.dots(x="year", y="population").fit_formula("population = a * exp(b * (year - 2000))")')
        self._check_layer_now(word, {"dots"})
        return self._add(word, "extra", _draw_fit_formula, formula=formula, start=start, between=between,
                         show_equation=bool(show_equation), colour=colour, thickness=thickness, said={})

    @friendly
    def fit_distribution(self, distribution="normal", colour=None, thickness=None):
        """The best-fitting distribution's curve over a histogram or density chart. chart.fit_distribution("normal")

        After a histogram or density, for example chart.histogram(x="marks", show="density")
        (with sdata.example("marks")).

        The parameters are found by maximum likelihood, as table.fit_distribution (R
        MASS::fitdistr()), and printed (once). Each colour group gets its own curve. Over a
        histogram the curve is scaled to what the bars show (count, percent or density).
        distribution= is any name of sdata.distribution, such as "gamma" or "lognormal".
        colour="red" makes every curve red; thickness=3 draws thicker curves.
        """
        word = "fit_distribution"
        thickness = check_thickness(word, thickness)
        _check_colour(word, colour)
        if not isinstance(distribution, str):
            raise SimplydataError('.fit_distribution() needs the name of a distribution, such as "normal" or "gamma".')
        from .._core import PHASES

        draws = [step.name for step in self._steps if step.phase == PHASES["draw"]]
        if draws and draws[-1] not in ("histogram", "density"):
            raise _histogram_layer_error(draws[-1])
        return self._add(word, "extra", _draw_fit_distribution, distribution=distribution, colour=colour,
                         thickness=thickness, said={})


# ====================================================================== helpers for lines() and qq_dots()
def draw_joined_lines(ctx, table, n, estimator, joined_by, dots, dashed, transparency, thickness, colours):
    """lines(joined_by=...) with a curve: each group's points (one per x, by show=) joined by that curve."""
    from ..table.between import join_one_direction, unique_mean

    ax = ctx.ax
    x, y, group = n["x"], n["y"], n.get("colour")
    for role, column in (("x", x), ("y", y)):
        series = table[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise SimplydataError(f'.lines(joined_by="{joined_by}") needs numbers on {role}, but "{column}" does not '
                                  'hold numbers. Use joined_by="straight lines" there.')
    parts = [(None, table)] if group is None else [(g, table[table[group] == g]) for g in pd.unique(table[group].dropna())]
    for (name, part), colour in zip(parts, colours):
        part = part[[x, y]].dropna()
        if part.empty:
            continue
        points = part.groupby(x, sort=True)[y].agg(estimator)
        px, py, _ = unique_mean(points.index.to_numpy(dtype=float), points.to_numpy(dtype=float))
        style = dict(color=colour, linestyle="--" if dashed else "-", linewidth=thickness, zorder=2)
        if transparency is not None:
            style["alpha"] = transparency
        label = None if name is None else str(name)
        if len(px) < 2:
            ax.plot(px, py, "o", color=colour, label=label)
            continue
        xs = np.union1d(np.linspace(px[0], px[-1], 400), px)
        ys = join_one_direction(px, py, xs, joined_by)
        ax.plot(xs, ys, label=label, **style)
        if dots:
            ax.plot(px, py, "o", color=colour, alpha=style.get("alpha"), zorder=2.1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if group is not None:
        ax.legend(title=group)


def lines_colours(ctx, table, group, fixed):
    from ._helpers import default_colour

    if group is None:
        return [default_colour(ctx.settings, fixed)]
    return palette_for(table, group, ctx.settings)


class _Quantiles:
    """What scipy.stats.probplot needs from a distribution: ppf()."""

    def __init__(self, dist):
        self.dist = dist

    def ppf(self, shares):
        return quantiles(self.dist, shares)


def qq_against(values, distribution):
    """probplot of the values against the fitted distribution: ((theory, ordered), (slope, intercept, r)), label."""
    from scipy import stats

    dist, printed = fit_distribution_to(values, distribution)
    inner = _frozen(dist)
    return stats.probplot(values, dist=inner if inner is not None else _Quantiles(dist)), dist, printed
