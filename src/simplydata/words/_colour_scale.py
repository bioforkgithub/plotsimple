"""The colour scale of heatmaps, contour maps, surfaces and dots coloured by a number column.

chart.colour_scale(...) only records what the user asked for (checked straight away);
scale_for() turns it into a matplotlib colour map and norm when a chart type is drawn:

* the map: a plain name ("blue to red"), any matplotlib/seaborn colour map name
  ("magma", "Blues", "rocket"), or two or more colours joined smoothly
  (LinearSegmentedColormap.from_list)
* between=[low, high]: Normalize(low, high); values outside get the end colours, and the
  colour bar shows arrows (extend=)
* log=True: LogNorm
* steps=5 or steps=[0, 5, 10, 20]: BoundaryNorm (one colour per step)
* middle=0: the middle colour of a diverging map sits at that value. Without between=,
  the scale goes as far on both sides of the middle (as matplotlib's CenteredNorm), so
  one unit is the same change of colour on both sides; with between=, TwoSlopeNorm.
"""

from __future__ import annotations

import numbers

import matplotlib
import numpy as np
from matplotlib import colors as mcolors

from .._errors import SimplydataError, suggest

# Plain names. Every one here is safe for colour-blind readers (ColorBrewer's colour-blind-safe
# diverging maps, the single-hue maps and the perceptually uniform maps of matplotlib).
PLAIN_NAMES = {
    "blue to red": "RdBu_r",
    "red to blue": "RdBu",
    "blue to yellow to red": "RdYlBu_r",
    "red to yellow to blue": "RdYlBu",
    "purple to orange": "PuOr_r",
    "orange to purple": "PuOr",
    "brown to green": "BrBG",
    "green to brown": "BrBG_r",
    "pink to green": "PiYG",
    "purple to green": "PRGn",
    "blues": "Blues",
    "greens": "Greens",
    "greys": "Greys",
    "grays": "Greys",
    "oranges": "Oranges",
    "reds": "Reds",
    "purples": "Purples",
    "viridis": "viridis",
    "cividis": "cividis",
    "magma": "magma",
    "inferno": "inferno",
    "plasma": "plasma",
}
SAFE_EXAMPLES = ["viridis", "cividis", "magma", "blue to red", "purple to orange", "blues", "greens"]
DEFAULT_MAP = "viridis"          # numbers going one way
DEFAULT_DIVERGING = "RdBu_r"     # numbers on both sides of a middle (blue below, red above)


def _registry():
    import seaborn  # noqa: F401  (registers rocket, mako, icefire, vlag ...)

    return {name.lower(): name for name in matplotlib.colormaps}


def resolve_map(names, word="colour_scale"):
    """names (a tuple of strings) -> (Colormap or None, a short name to show)."""
    if not names:
        return None, None
    for n in names:
        if not isinstance(n, str):
            raise SimplydataError(
                f".{word}() needs the name of a colour scale or colour names, not {n!r}.\n"
                f'For example chart.{word}("magma") or chart.{word}("white", "navy").')
    if len(names) == 1:
        name = names[0]
        key = " ".join(name.lower().replace("-", " ").split())
        maps = _registry()
        if key in PLAIN_NAMES:
            return matplotlib.colormaps[PLAIN_NAMES[key]], name
        if name.lower() in maps:
            return matplotlib.colormaps[maps[name.lower()]], name
        if mcolors.is_color_like(name):
            raise SimplydataError(
                f'.{word}() needs at least two colours to go between, but got only "{name}".\n'
                f'For example chart.{word}("white", "{name}"), or a scale name such as "magma".')
        guess = suggest(key, list(PLAIN_NAMES) + sorted(maps), n=1)
        lines = [f'.{word}() does not know the colour scale "{name}".']
        if guess:
            shown = guess[0] if guess[0] in PLAIN_NAMES else maps[guess[0]]
            lines.append(f'Did you mean "{shown}"?')
        lines.append("Colour-blind-safe choices: " + ", ".join(f'"{n}"' for n in SAFE_EXAMPLES) + ".")
        lines.append("Any matplotlib colour map name works too (such as \"Blues\" or \"turbo\"), "
                     f'or give two or more colours: chart.{word}("white", "navy").')
        raise SimplydataError("\n".join(lines))
    bad = [n for n in names if not mcolors.is_color_like(n)]
    if bad:
        raise SimplydataError(
            f'.{word}() got several names, so it reads them as colours, but "{bad[0]}" is not a colour.\n'
            f'For example chart.{word}("white", "navy") or chart.{word}("blue", "white", "red").')
    return mcolors.LinearSegmentedColormap.from_list(" to ".join(names), list(names)), " to ".join(names)


def _number(v):
    return not isinstance(v, bool) and isinstance(v, numbers.Real) and np.isfinite(v)


def check_options(middle, log, steps, between, word="colour_scale"):
    """Check the options straight away. Returns (middle, log, steps, between) cleaned."""
    if middle is not None and not _number(middle):
        raise SimplydataError(f"middle= in .{word}() should be a number, such as middle=0, not {middle!r}.")
    if not isinstance(log, bool):
        raise SimplydataError(f"log= in .{word}() should be True or False, not {log!r}.")
    if between is not None:
        if isinstance(between, str) or not hasattr(between, "__len__") or len(between) != 2 \
                or not all(_number(v) for v in between):
            raise SimplydataError(
                f".{word}() needs two numbers in between=, the low end and the high end.\n"
                f"For example chart.{word}(between=[0, 30]).")
        between = [float(between[0]), float(between[1])]
        if not between[0] < between[1]:
            raise SimplydataError(
                f".{word}(between={between!r}): the low end must be smaller than the high end.")
    if steps is not None:
        if isinstance(steps, bool):
            ok = False
        elif isinstance(steps, (int, np.integer)):
            ok = 2 <= steps <= 50
        else:
            ok = (hasattr(steps, "__len__") and not isinstance(steps, str) and len(steps) >= 2
                  and all(_number(v) for v in steps)
                  and all(a < b for a, b in zip(list(steps)[:-1], list(steps)[1:])))
        if not ok:
            raise SimplydataError(
                f"steps= in .{word}() should be how many colours (a whole number from 2 to 50, such as steps=5)\n"
                f"or the edges where each colour starts, going up (such as steps=[0, 5, 10, 20]), not {steps!r}.")
        steps = int(steps) if isinstance(steps, (int, np.integer)) else [float(v) for v in steps]
    if log:
        if middle is not None:
            raise SimplydataError(f".{word}() cannot use log=True and middle= together. Choose one.")
        if between is not None and between[0] <= 0:
            raise SimplydataError(
                f".{word}(log=True) cannot show 0 or negative numbers, but between= starts at {between[0]:g}.\n"
                "Start above 0, for example between=[1, 1000].")
        if isinstance(steps, list) and steps[0] <= 0:
            raise SimplydataError(f".{word}(log=True) cannot have steps at 0 or below; start the steps above 0.")
    if steps is not None and middle is not None:
        raise SimplydataError(
            f".{word}() cannot use steps= and middle= together: the steps already say where each colour goes.\n"
            "Put the middle among the steps instead, for example steps=[-10, -5, 0, 5, 10].")
    if middle is not None and between is not None and not between[0] < middle < between[1]:
        raise SimplydataError(
            f".{word}(): middle={middle:g} should lie between the two ends of between={between!r}.")
    return middle, log, steps, between


def wanted(settings) -> bool:
    return bool(settings.get("colour_scale"))


def scale_for(ctx, values, word, diverge_at_zero=False):
    """(cmap, norm, extend, levels) for these values, from chart.colour_scale (or the defaults).

    diverge_at_zero: with no colour_scale, values on both sides of 0 get "blue to red" with
    white at 0 (the heatmap's usual look).
    levels: the edges for a filled contour map (None: let matplotlib choose 10).
    """
    spec = ctx.settings.get("colour_scale") or {}
    if spec:
        ctx.store["colour_scale_used"] = True
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise SimplydataError(f".{word}() found no numbers to colour.")
    lo_data, hi_data = float(values.min()), float(values.max())
    cmap, middle, log = spec.get("map"), spec.get("middle"), spec.get("log", False)
    steps, between = spec.get("steps"), spec.get("between")
    if middle is None and cmap is None and diverge_at_zero and not log and steps is None:
        lo, hi = between or (lo_data, hi_data)
        if lo < 0 < hi:
            middle = 0.0
    if cmap is None:
        cmap = matplotlib.colormaps[DEFAULT_DIVERGING if middle is not None else DEFAULT_MAP]
    lo, hi = between or (lo_data, hi_data)
    levels = None
    if log:
        if lo_data <= 0:
            raise SimplydataError(
                f".colour_scale(log=True) cannot colour zero or negative numbers, and .{word}() has some "
                f"(smallest: {lo_data:g}).\nA log scale only works when every value is bigger than 0. "
                "Leave out log=True, or remove those values from the table first.")
        norm = mcolors.LogNorm(lo, hi)
        from matplotlib.ticker import LogLocator

        ticks = [t for t in LogLocator().tick_values(lo, hi) if lo <= t <= hi]
        levels = np.array(ticks) if len(ticks) >= 3 else np.geomspace(lo, hi, 11)
        if levels[0] > lo:
            levels = np.r_[lo, levels]
        if levels[-1] < hi:
            levels = np.r_[levels, hi]
    elif steps is not None:
        if isinstance(steps, int):
            if not lo < hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, steps + 1)
            # the top step holds the biggest value too (a step's top edge belongs to the next one)
            edges[-1] = np.nextafter(edges[-1], np.inf)
        else:
            edges = np.asarray(steps, dtype=float)
        levels = edges
        norm = mcolors.BoundaryNorm(edges, cmap.N, extend=_extend(lo_data, hi_data, edges[0], edges[-1]))
    elif middle is not None:
        if between is not None:
            norm = mcolors.TwoSlopeNorm(vcenter=middle, vmin=lo, vmax=hi)
        else:
            half = max(hi - middle, middle - lo) or 1.0
            norm = mcolors.Normalize(middle - half, middle + half)
    else:
        if not lo < hi:  # every value the same: one colour in the middle of the scale
            lo, hi = lo - 0.5, hi + 0.5
        norm = mcolors.Normalize(lo, hi)
        if between is not None:
            levels = np.linspace(lo, hi, 11)
    extend = _extend(lo_data, hi_data, norm.vmin, norm.vmax)
    return cmap, norm, extend, levels


def _extend(lo_data, hi_data, low, high):
    below, above = lo_data < low, hi_data > high
    return "both" if below and above else "min" if below else "max" if above else "neither"


def check_used(ctx):
    """A colour scale on a chart that has no colours for numbers is a mistake worth telling."""
    if ctx.settings.get("colour_scale") and not ctx.store.get("colour_scale_used"):
        raise SimplydataError(
            ".colour_scale() sets the colours for numbers, but this chart has nothing coloured by numbers.\n"
            "It works with chart.heatmap(), chart.contour(), chart.surface(), and chart.dots() or "
            'chart.dots_3d() with colour= a number column, for example chart.dots(x="hours", y="marks", '
            'colour="height").\nFor groups, use chart.colours("blue", "orange") instead.')
