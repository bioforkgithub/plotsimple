"""Chart types: what you draw.

Calling two chart types on the same chart puts them on top of each other.
"""

from __future__ import annotations

import seaborn as sns

from .._data import frame_for
from .._errors import PlotsimpleError, friendly
from .._stats import SHOW, check_error, check_show
from .._style import COLOURS
from ._helpers import check, colour_role, palette_for, single_list


def _remember(ctx, kind, table, names, **more):
    ctx.last = dict(kind=kind, table=table, x=names.get("x"), y=names.get("y"),
                    colour=names.get("colour"), **more)


# ---------------------------------------------------------------- bars
def _draw_bars(ctx, x, y, colour, show, error, stacked, sideways):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, colour=group)
    how = SHOW[show]
    ax = ctx.ax

    if stacked:
        cols = [n["x"], n["colour"]]
        if how == "count":
            wide = table.groupby(cols, sort=False).size().unstack(n["colour"])
        else:
            wide = table.groupby(cols, sort=False)[n["y"]].agg(how).unstack(n["colour"])
        wide.plot(kind="barh" if sideways else "bar", stacked=True, ax=ax, rot=0, width=0.8,
                  color=palette_for(table, n["colour"]))
        ax.legend(title=n["colour"])
        if sideways:
            ax.set_xlabel(n["y"] or "count")
        else:
            ax.set_ylabel(n["y"] or "count")
    else:
        kw = dict(data=table, hue=n["colour"], ax=ax, saturation=1)  # exact colours, as asked
        if n["colour"]:
            kw["palette"] = palette_for(table, n["colour"])
        else:
            kw["color"] = fixed or COLOURS[0]
        if how == "count":
            kw.update(y=n["x"]) if sideways else kw.update(x=n["x"])
            sns.countplot(**kw)
        else:
            if sideways:
                kw.update(x=n["y"], y=n["x"], orient="h")
            else:
                kw.update(x=n["x"], y=n["y"])
            kw["estimator"] = how
            kw["errorbar"] = check_error("bars", error, show)
            if error:
                kw["capsize"] = 0.15
            sns.barplot(**kw)
    _remember(ctx, "bars", table, n, sideways=sideways)


# ---------------------------------------------------------------- lines
def _draw_lines(ctx, x, y, colour, show, error, dots, dashed):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, colour=group)
    kw = dict(data=table, x=n["x"], y=n["y"], hue=n["colour"], ax=ctx.ax,
              estimator=SHOW[show], errorbar=check_error("lines", error, show))
    if n["colour"]:
        kw["palette"] = palette_for(table, n["colour"])
    else:
        kw["color"] = fixed or COLOURS[0]
    if dots:
        kw["marker"] = "o"
    if dashed:
        kw["linestyle"] = "--"
    sns.lineplot(**kw)
    _remember(ctx, "lines", table, n)


# ---------------------------------------------------------------- dots
def _draw_dots(ctx, x, y, colour, size, shape):
    group, fixed = colour_role(ctx.data, colour)
    size_col = size if not isinstance(size, (int, float)) else None
    table, n = frame_for(ctx.data, x=x, y=y, colour=group, size=size_col, shape=shape)
    kw = dict(data=table, x=n["x"], y=n["y"], hue=n["colour"], size=n["size"], style=n["shape"], ax=ctx.ax)
    if n["colour"]:
        kw["palette"] = palette_for(table, n["colour"])
    else:
        kw["color"] = fixed or COLOURS[0]
    if isinstance(size, (int, float)):
        kw["s"] = size ** 2
    sns.scatterplot(**kw)
    _remember(ctx, "dots", table, n)


class ChartWords:
    @friendly
    def bars(self, x=None, y=None, colour=None, show="average", error=None,
             stacked=False, sideways=False):
        """Bars. chart.bars(x="name", y="marks")

        colour="site" gives each group its own colour, or colour="blue" one colour.
        show="average" (default), "median", "total" or "count".
        error="sd", "se" or "ci" adds error bars (with show="average").
        stacked=True puts groups on top of each other; sideways=True lays bars down.
        """
        check_show("bars", show)
        check_error("bars", error, show)
        x, y = single_list(x, y)
        if x is None:
            raise PlotsimpleError('.bars() needs x=, for example chart.bars(x="name", y="marks").')
        if y is None and SHOW[show] != "count":
            raise PlotsimpleError(
                '.bars() needs y=, for example chart.bars(x="name", y="marks").\n'
                'To count rows instead, use chart.bars(x="name", show="count").'
            )
        group, _ = colour_role(self._data, colour)
        if stacked and group is None:
            raise PlotsimpleError('stacked=True needs groups to stack: add colour="a column".')
        if stacked and error:
            raise PlotsimpleError("Error bars cannot be drawn on stacked bars. Remove error= or stacked=True.")
        check(self._data, x=x, y=y, colour=group)
        return self._add("bars", "draw", _draw_bars, x=x, y=y, colour=colour, show=show,
                         error=error, stacked=stacked, sideways=sideways)

    @friendly
    def lines(self, x=None, y=None, colour=None, show="average", error=None, dots=False, dashed=False):
        """Lines. chart.lines(x="day", y="temperature")

        colour="city" draws one line per group. dots=True marks each point.
        dashed=True draws a dashed line. error="sd", "se" or "ci" adds a band.
        """
        check_show("lines", show)
        if SHOW[show] == "count":
            raise PlotsimpleError('.lines() cannot use show="count". Use "average", "median" or "total".')
        check_error("lines", error, show)
        x, y = single_list(x, y)
        if x is None or y is None:
            raise PlotsimpleError('.lines() needs x= and y=, for example chart.lines(x="day", y="temperature").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        return self._add("lines", "draw", _draw_lines, x=x, y=y, colour=colour, show=show,
                         error=error, dots=dots, dashed=dashed)

    @friendly
    def dots(self, x=None, y=None, colour=None, size=None, shape=None):
        """Dots (a scatter chart). chart.dots(x="height", y="weight")

        colour="group" colours dots by group. size="age" makes dots bigger for bigger
        values, or size=10 makes all dots that big. shape="type" gives each group a shape.
        """
        x, y = single_list(x, y)
        if x is None or y is None:
            raise PlotsimpleError('.dots() needs x= and y=, for example chart.dots(x="height", y="weight").')
        group, _ = colour_role(self._data, colour)
        size_col = size if not isinstance(size, (int, float)) else None
        check(self._data, x=x, y=y, colour=group, size=size_col, shape=shape)
        return self._add("dots", "draw", _draw_dots, x=x, y=y, colour=colour, size=size, shape=shape)

    scatter = dots  # standard name, same word
