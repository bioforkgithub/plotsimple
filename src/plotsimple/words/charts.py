"""Chart types: what you draw.

Calling two chart types on the same chart puts them on top of each other.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import seaborn as sns

from .._core import PHASES
from .._data import frame_for
from .._errors import PlotsimpleError, friendly
from .._stats import SHOW, check_error, check_show
from ._helpers import check, colour_role, default_colour, palette_for, single_list

# all_points: up to this many points in one group, dots are packed side by side (a swarm);
# above it they are spread at random (a strip), because a swarm would not fit.
SWARM_LIMIT = 100
JITTER_SEED = 0  # the same "random" spread every time the chart is drawn


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
                  color=palette_for(table, n["colour"], ctx.settings))
        ax.legend(title=n["colour"])
        if sideways:
            ax.set_xlabel(n["y"] or "count")
        else:
            ax.set_ylabel(n["y"] or "count")
    else:
        kw = dict(data=table, hue=n["colour"], ax=ax, saturation=1)  # exact colours, as asked
        if n["colour"]:
            kw["palette"] = palette_for(table, n["colour"], ctx.settings)
        else:
            kw["color"] = default_colour(ctx.settings, fixed)
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
        kw["palette"] = palette_for(table, n["colour"], ctx.settings)
    else:
        kw["color"] = default_colour(ctx.settings, fixed)
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
        kw["palette"] = palette_for(table, n["colour"], ctx.settings)
    else:
        kw["color"] = default_colour(ctx.settings, fixed)
    if isinstance(size, (int, float)):
        kw["s"] = size ** 2
    sns.scatterplot(**kw)
    _remember(ctx, "dots", table, n)


# ---------------------------------------------------------------- shared helpers
def _seaborn_colours(kw, table, n, ctx, fixed):
    """Add palette= (groups) or color= (one colour) to a seaborn call."""
    if n.get("colour"):
        kw["palette"] = palette_for(table, n["colour"], ctx.settings)
    else:
        kw["color"] = default_colour(ctx.settings, fixed)
    return kw


def _groups(table, group):
    """[(name, rows)] in the order the groups first appear; [(None, table)] without groups."""
    if group is None:
        return [(None, table)]
    return [(g, table[table[group] == g]) for g in pd.unique(table[group].dropna())]


def _group_colours(table, group, ctx, fixed):
    if group is None:
        return [default_colour(ctx.settings, fixed)]
    return palette_for(table, group, ctx.settings)


def _ordered(values):
    """Numbers and dates are sorted; words keep the order they appear in."""
    return pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values)


def _is_number_column(col):
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def _check_numbers(data, word, **roles):
    """Columns that must hold numbers: say so on the user's line, not while drawing."""
    for role, value in roles.items():
        if isinstance(value, str) and value in data.columns and not _is_number_column(data[value]):
            raise PlotsimpleError(
                f'.{word}() needs numbers for {role}=, but column "{value}" holds words or other things.\n'
                "Pick a column with numbers in it."
            )


def _nice_number(v) -> str:
    """Numbers written on a heatmap: no 1.2e+02, just 123 or 12.3 or 0.123."""
    if v is None or pd.isna(v):
        return ""
    v = float(v)
    if v.is_integer() and abs(v) < 1e7:
        return str(int(v))
    digits = max(0, min(3, 2 - math.floor(math.log10(abs(v)))))
    text = f"{v:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


@contextmanager
def _quiet():
    """Hide library notes about their own future changes; beginners cannot act on them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        yield


def _colour_bar(ctx, mappable, label=None, **kw):
    bar = ctx.fig.colorbar(mappable, ax=ctx.ax, **kw)
    if label:
        bar.set_label(str(label))
    ctx.store["colour_bar"] = bar
    return bar


def _to_grid(table, x, y, z, word):
    """x, y, z columns -> 2-D grids X, Y, Z.

    A full grid (every x with every y) is used as it is. Scattered points are joined into
    a 100 x 100 grid with straight-line (linear) interpolation from SciPy.
    Several values at the same x and y are averaged.
    """
    t = table[[x, y, z]].dropna()
    t = t.groupby([x, y], as_index=False)[z].mean()
    xs, ys = np.sort(t[x].unique()), np.sort(t[y].unique())
    if len(xs) < 2 or len(ys) < 2:
        raise PlotsimpleError(f".{word}() needs at least 2 different x values and 2 different y values.")
    if len(xs) * len(ys) == len(t):
        wide = t.pivot(index=y, columns=x, values=z).loc[ys, xs]
        gx, gy = np.meshgrid(xs, ys)
        return gx, gy, wide.to_numpy(dtype=float)
    from scipy.interpolate import griddata

    gx, gy = np.meshgrid(np.linspace(xs[0], xs[-1], 100), np.linspace(ys[0], ys[-1], 100))
    try:
        gz = griddata((t[x].to_numpy(float), t[y].to_numpy(float)), t[z].to_numpy(float),
                      (gx, gy), method="linear")
    except Exception:
        raise PlotsimpleError(
            f".{word}() could not join your points into a surface.\n"
            "It needs at least 3 points that do not all lie on one straight line."
        ) from None
    return gx, gy, gz


# ---------------------------------------------------------------- area
def _draw_area(ctx, x, y, colour, stacked):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, colour=group)
    ax = ctx.ax
    keys = [n["x"]] + ([n["colour"]] if n["colour"] else [])
    # several values at the same x are averaged (like .lines())
    mean = table.groupby(keys, sort=False)[n["y"]].mean()
    colours = _group_colours(table, n["colour"], ctx, fixed)
    if n["colour"]:
        wide = mean.unstack(n["colour"])
        wide = wide[[g for g in pd.unique(table[n["colour"]].dropna()) if g in wide.columns]]
    else:
        wide = mean.to_frame(n["y"])
    if _ordered(table[n["x"]]):
        wide = wide.sort_index()
    xs = list(wide.index)
    if stacked:
        ax.stackplot(xs, *[wide[c].fillna(0).to_numpy(dtype=float) for c in wide.columns],
                     labels=[str(c) for c in wide.columns], colors=colours, alpha=0.85)
    else:
        for c, col in zip(colours, wide.columns):
            ys = wide[col].to_numpy(dtype=float)
            ax.fill_between(xs, ys, alpha=0.35, color=c, linewidth=0)
            ax.plot(xs, ys, color=c, label=str(col) if n["colour"] else None)
    if n["colour"]:
        ax.legend(title=n["colour"])
    ax.set_xlabel(n["x"])
    ax.set_ylabel(n["y"])
    ctx.store["area_table"] = wide  # the heights drawn, one column per group
    _remember(ctx, "area", table, n)


# ---------------------------------------------------------------- histogram
def _draw_histogram(ctx, x, bins, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, colour=group)
    kw = dict(data=table, x=n["x"], hue=n["colour"], bins=bins, ax=ctx.ax)
    sns.histplot(**_seaborn_colours(kw, table, n, ctx, fixed))
    ctx.ax.set_ylabel("count")
    _remember(ctx, "histogram", table, n)


# ---------------------------------------------------------------- boxes and violins
def _draw_boxes(ctx, x, y, colour, violins):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, colour=group)
    kw = dict(data=table, x=n["x"], y=n["y"], hue=n["colour"], ax=ctx.ax, saturation=1)
    _seaborn_colours(kw, table, n, ctx, fixed)
    if n["colour"] and n["colour"] not in (n["x"], n["y"]):
        kw["gap"] = 0.1  # a little space between the boxes of one group
    if violins:
        # cut=0: a violin stops at the smallest and largest value, so it never shows
        # values that are impossible (such as marks below 0)
        sns.violinplot(cut=0, **kw)
    else:
        with _quiet():
            sns.boxplot(**kw)
    _remember(ctx, "violins" if violins else "boxes", table, n)


# ---------------------------------------------------------------- all points
def _draw_all_points(ctx, x, y, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, colour=group)
    dodge = bool(n["colour"]) and n["colour"] not in (n["x"], n["y"])
    kw = dict(data=table, x=n["x"], y=n["y"], hue=n["colour"], ax=ctx.ax, dodge=dodge)
    _seaborn_colours(kw, table, n, ctx, fixed)
    keys = [k for k in (n["x"], n["y"]) if k is not None and not _is_number_column(table[k])]
    if dodge:
        keys.append(n["colour"])
    biggest = table.groupby(keys).size().max() if keys else len(table)
    if biggest <= SWARM_LIMIT:
        sns.swarmplot(**kw)
    else:
        # seaborn spreads strip points with NumPy's shared random numbers; fix them so
        # the chart is the same every time, and put the user's random state back after
        state = np.random.get_state()
        np.random.seed(JITTER_SEED)
        try:
            sns.stripplot(alpha=0.6, **kw)
        finally:
            np.random.set_state(state)
    _remember(ctx, "all_points", table, n)


# ---------------------------------------------------------------- density
def _draw_density(ctx, x, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, colour=group)
    # common_norm=False: each group's curve has area 1, so small groups are not squashed
    kw = dict(data=table, x=n["x"], hue=n["colour"], ax=ctx.ax, common_norm=False)
    sns.kdeplot(**_seaborn_colours(kw, table, n, ctx, fixed))
    _remember(ctx, "density", table, n)


# ---------------------------------------------------------------- heatmap
def _heatmap_table(data, x, y, value):
    if value is not None:
        table, n = frame_for(data, x=x, y=y, value=value)
        wide = table.pivot_table(index=n["y"], columns=n["x"], values=n["value"],
                                 aggfunc="mean", sort=False)
        if _ordered(table[n["y"]]):
            wide = wide.sort_index(axis=0)
        if _ordered(table[n["x"]]):
            wide = wide.sort_index(axis=1)
        return wide, n["value"]
    numbers = [c for c in data.columns if _is_number_column(data[c])]
    words = [c for c in data.columns
             if c not in numbers and not pd.api.types.is_bool_dtype(data[c])
             and not pd.api.types.is_datetime64_any_dtype(data[c])]
    wide = data[numbers]
    if words:
        wide = wide.set_axis(data[words[0]].astype(str), axis=0)
        wide.index.name = words[0]
    return wide, None


def _cluster_order(values):
    from scipy.cluster import hierarchy

    # the same choices as seaborn's clustermap: average linkage, straight-line distance
    tree = hierarchy.linkage(values, method="average", metric="euclidean")
    return tree, hierarchy.leaves_list(tree)


def _draw_tree(ax, tree, orientation, n_leaves):
    from scipy.cluster import hierarchy

    hierarchy.dendrogram(tree, ax=ax, orientation=orientation, no_labels=True,
                         link_color_func=lambda _: "#444444")
    if orientation == "left":
        ax.set_ylim(10 * n_leaves, 0)  # the first heatmap row is at the top
    else:
        ax.set_xlim(0, 10 * n_leaves)
    ax.set_axis_off()


def _draw_heatmap(ctx, x, y, value, numbers, cluster):
    wide, label = _heatmap_table(ctx.data, x, y, value)
    if wide.empty:
        raise PlotsimpleError(".heatmap() found no numbers to draw.")
    ax = ctx.ax
    trees = None
    if cluster:
        if wide.isna().any().any():
            raise PlotsimpleError(
                "cluster=True cannot sort rows and columns that have empty cells.\n"
                "Fill or remove the empty cells first, or use cluster=False."
            )
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            raise PlotsimpleError("cluster=True needs at least 2 rows and 2 columns.")
        values = wide.to_numpy(dtype=float)
        row_tree, row_order = _cluster_order(values)
        col_tree, col_order = _cluster_order(values.T)
        wide = wide.iloc[row_order, col_order]
        trees = (row_tree, col_tree)
    lo, hi = np.nanmin(wide.to_numpy(dtype=float)), np.nanmax(wide.to_numpy(dtype=float))
    kw = dict(ax=ax, cbar=False, cmap="viridis")
    if lo < 0 < hi:  # values on both sides of 0: 0 is white, below is blue, above is red
        kw.update(cmap="RdBu_r", center=0)
    if numbers:
        kw.update(annot=wide.map(_nice_number).to_numpy(), fmt="")
    with _quiet():
        sns.heatmap(wide, **kw)
    ax.grid(False)
    ax.tick_params(axis="y", rotation=0)
    if trees:
        n_rows, n_cols = wide.shape
        ax.yaxis.tick_right()  # the row tree takes the left side
        ax.yaxis.set_label_position("right")
        _draw_tree(ax.inset_axes([-0.22, 0, 0.2, 1]), trees[0], "left", n_rows)
        _draw_tree(ax.inset_axes([0, 1.02, 1, 0.2]), trees[1], "top", n_cols)
    _colour_bar(ctx, ax.collections[0], label)
    ctx.store["heatmap_table"] = wide  # the numbers drawn, in the order drawn
    ctx.last = dict(kind="heatmap", table=wide, x=None, y=None, colour=None)


# ---------------------------------------------------------------- pie
def _draw_pie(ctx, labels, values):
    table, n = frame_for(ctx.data, labels=labels, values=values)
    if n["values"] is None:
        sizes = table[n["labels"]].value_counts(sort=False, dropna=True)
        sizes = sizes.reindex(pd.unique(table[n["labels"]].dropna()))
    else:
        sizes = table.groupby(n["labels"], sort=False)[n["values"]].sum()
    if (sizes < 0).any():
        raise PlotsimpleError(".pie() cannot show values below 0. A slice cannot be smaller than nothing.")
    if sizes.sum() == 0:
        raise PlotsimpleError(".pie() needs at least one value bigger than 0.")
    ax = ctx.ax
    ax.pie(sizes.to_numpy(dtype=float), labels=[str(i) for i in sizes.index],
           colors=palette_for(table, n["labels"], ctx.settings), autopct="%1.0f%%",
           startangle=90, counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.grid(False)
    ctx.store["pie_sizes"] = sizes
    ctx.last = dict(kind="pie", table=table, x=n["labels"], y=n["values"], colour=None)


# ---------------------------------------------------------------- polar
def _draw_polar(ctx, angle, distance, colour, degrees):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, angle=angle, distance=distance, colour=group)
    ax = ctx.ax
    for (name, rows), c in zip(_groups(table, n["colour"]), _group_colours(table, n["colour"], ctx, fixed)):
        theta = rows[n["angle"]].to_numpy(dtype=float)
        if degrees:
            theta = np.deg2rad(theta)
        ax.scatter(theta, rows[n["distance"]].to_numpy(dtype=float), color=c,
                   label=None if name is None else str(name), zorder=3)
    if n["colour"]:
        ax.legend(title=n["colour"], loc="upper left", bbox_to_anchor=(1.05, 1))
    _remember(ctx, "polar", table, {"x": n["angle"], "y": n["distance"], "colour": n["colour"]})


# ---------------------------------------------------------------- contour and surface
def _draw_contour(ctx, x, y, z, filled):
    table, n = frame_for(ctx.data, x=x, y=y, z=z)
    gx, gy, gz = _to_grid(table, n["x"], n["y"], n["z"], "contour")
    ax = ctx.ax
    if filled:
        shapes = ax.contourf(gx, gy, gz, levels=10, cmap="viridis")
        ax.contour(shapes, colors="white", linewidths=0.5, alpha=0.6, linestyles="solid")
    else:
        shapes = ax.contour(gx, gy, gz, levels=10, cmap="viridis")
        ax.clabel(shapes, fontsize="small")
    ax.grid(False)
    ax.set_xlabel(n["x"])
    ax.set_ylabel(n["y"])
    _colour_bar(ctx, shapes, n["z"])
    ctx.store["grid"] = (gx, gy, gz)
    _remember(ctx, "contour", table, n, z=n["z"])


def _draw_surface(ctx, x, y, z):
    table, n = frame_for(ctx.data, x=x, y=y, z=z)
    gx, gy, gz = _to_grid(table, n["x"], n["y"], n["z"], "surface")
    ax = ctx.ax
    surf = ax.plot_surface(gx, gy, np.ma.masked_invalid(gz), cmap="viridis", linewidth=0,
                           antialiased=True, vmin=np.nanmin(gz), vmax=np.nanmax(gz))
    ax.set_xlabel(n["x"])
    ax.set_ylabel(n["y"])
    ax.set_zlabel(n["z"])
    _colour_bar(ctx, surf, n["z"], shrink=0.6, pad=0.1)
    ctx.store["grid"] = (gx, gy, gz)
    _remember(ctx, "surface", table, n, z=n["z"])


def _draw_dots_3d(ctx, x, y, z, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = frame_for(ctx.data, x=x, y=y, z=z, colour=group)
    ax = ctx.ax
    g = n["colour"]
    if g is not None and _is_number_column(table[g]) and table[g].nunique() > 8:
        # a number column with many values: a colour scale, not separate groups
        pts = ax.scatter(table[n["x"]], table[n["y"]], table[n["z"]], c=table[g],
                         cmap="viridis", depthshade=False)
        _colour_bar(ctx, pts, g, shrink=0.6, pad=0.1)
    else:
        for (name, rows), c in zip(_groups(table, g), _group_colours(table, g, ctx, fixed)):
            ax.scatter(rows[n["x"]], rows[n["y"]], rows[n["z"]], color=c, depthshade=False,
                       label=None if name is None else str(name))
        if g:
            ax.legend(title=g)
    ax.set_xlabel(n["x"])
    ax.set_ylabel(n["y"])
    ax.set_zlabel(n["z"])
    _remember(ctx, "dots_3d", table, n, z=n["z"])


_KINDS = {"polar": "a round (polar) chart", "3d": "a 3D chart", None: "a flat chart"}


class ChartWords:
    def _need_axes(self, word, kind=None):
        """Polar and 3D charts need their own kind of axes; flat chart types cannot share them."""
        current = self._settings.get("projection")
        if kind == current:
            return
        drawn = list(dict.fromkeys(s.name for s in self._steps if s.phase == PHASES["draw"]))
        if drawn:
            others = ", ".join(f".{d}()" for d in drawn)
            raise PlotsimpleError(
                f".{word}() draws {_KINDS[kind]}, but this chart already has {others}, "
                f"which make{'s' if len(drawn) == 1 else ''} {_KINDS[current]}.\n"
                f"They cannot share one chart. Start a new chart for .{word}(), for example:\n"
                f"    other = ps.chart(table)\n"
                f"    other.{word}(...)"
            )
        self._set(projection=kind)

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
        self._need_axes("bars")
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
        self._need_axes("lines")
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
        self._need_axes("dots")
        return self._add("dots", "draw", _draw_dots, x=x, y=y, colour=colour, size=size, shape=shape)

    scatter = dots  # standard name, same word

    @staticmethod
    def _yes_no(word, **options):
        for option, value in options.items():
            if not isinstance(value, bool):
                raise PlotsimpleError(f"{option}= in .{word}() should be True or False, not {value!r}.")

    @friendly
    def area(self, x=None, y=None, colour=None, stacked=False):
        """A line with the space under it coloured in. chart.area(x="year", y="sales")

        colour="shop" gives each group its own area; stacked=True piles the groups on top
        of each other, so the top edge shows the total.
        """
        self._yes_no("area", stacked=stacked)
        x, y = single_list(x, y)
        if x is None or y is None:
            raise PlotsimpleError('.area() needs x= and y=, for example chart.area(x="year", y="sales").')
        group, _ = colour_role(self._data, colour)
        if stacked and group is None:
            raise PlotsimpleError('stacked=True needs groups to stack: add colour="a column".')
        check(self._data, x=x, y=y, colour=group)
        _check_numbers(self._data, "area", y=y)
        self._need_axes("area")
        return self._add("area", "draw", _draw_area, x=x, y=y, colour=colour, stacked=stacked)

    @friendly
    def histogram(self, x=None, bins=10, colour=None):
        """Bars that count how many values fall in each range. chart.histogram(x="marks", bins=10)

        bins= is how many bars (or a list of where each bar starts and ends).
        colour="class" draws one see-through histogram per group.
        """
        if x is None:
            raise PlotsimpleError('.histogram() needs x=, for example chart.histogram(x="marks").')
        if isinstance(bins, bool) or not (
            (isinstance(bins, (int, np.integer)) and bins >= 1)
            or (isinstance(bins, (list, tuple, np.ndarray)) and len(bins) >= 2
                and all(isinstance(b, (int, float, np.number)) for b in bins)
                and all(a < b for a, b in zip(bins[:-1], bins[1:])))
        ):
            raise PlotsimpleError(
                f"bins= should be a whole number of bars (such as bins=10) or a list of edges\n"
                f"going up (such as bins=[0, 50, 70, 100]), not {bins!r}."
            )
        bins = bins if isinstance(bins, (int, np.integer)) else [float(b) for b in bins]
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, colour=group)
        _check_numbers(self._data, "histogram", x=x)
        self._need_axes("histogram")
        return self._add("histogram", "draw", _draw_histogram, x=x, bins=bins, colour=colour)

    def _boxes_or_violins(self, word, x, y, colour):
        if x is None and y is None:
            raise PlotsimpleError(
                f'.{word}() needs y= (and usually x=), for example chart.{word}(x="class", y="marks").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        self._need_axes(word)
        return self._add(word, "draw", _draw_boxes, x=x, y=y, colour=colour, violins=word == "violins")

    @friendly
    def boxes(self, x=None, y=None, colour=None):
        """Boxes that show the middle half of the values and the median. chart.boxes(x="class", y="marks")

        The line in the box is the median, the box holds the middle half of the values,
        the whiskers reach the rest, and lone dots are values far away from the others.
        """
        return self._boxes_or_violins("boxes", x, y, colour)

    @friendly
    def violins(self, x=None, y=None, colour=None):
        """Shapes that are wide where many values are. chart.violins(x="class", y="marks")

        Each violin stops at the smallest and the largest value; the small box inside
        shows the middle half of the values.
        """
        return self._boxes_or_violins("violins", x, y, colour)

    @friendly
    def count_bars(self, x=None, colour=None, sideways=False):
        """Bars that count how many rows each group has. chart.count_bars(x="fruit")

        colour="shop" splits each bar by group; sideways=True lays the bars down.
        """
        self._yes_no("count_bars", sideways=sideways)
        if x is None:
            raise PlotsimpleError('.count_bars() needs x=, for example chart.count_bars(x="fruit").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, colour=group)
        self._need_axes("count_bars")
        return self._add("count_bars", "draw", _draw_bars, x=x, y=None, colour=colour, show="count",
                         error=None, stacked=False, sideways=sideways)

    @friendly
    def all_points(self, x=None, y=None, colour=None):
        """Every value as a dot, spread out so no dot hides another. chart.all_points(x="class", y="marks")

        Good on top of .boxes() or .violins(), to show each value as well.
        """
        if x is None and y is None:
            raise PlotsimpleError(
                '.all_points() needs y= (and usually x=), for example chart.all_points(x="class", y="marks").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        self._need_axes("all_points")
        return self._add("all_points", "draw", _draw_all_points, x=x, y=y, colour=colour)

    @friendly
    def density(self, x=None, colour=None):
        """A smooth curve that is high where many values are. chart.density(x="marks")

        colour="class" draws one curve per group; each curve has an area of 1.
        """
        if x is None:
            raise PlotsimpleError('.density() needs x=, for example chart.density(x="marks").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, colour=group)
        _check_numbers(self._data, "density", x=x)
        self._need_axes("density")
        return self._add("density", "draw", _draw_density, x=x, colour=colour)

    @friendly
    def heatmap(self, x=None, y=None, value=None, numbers=False, cluster=False):
        """A grid of coloured squares, one colour for each number. chart.heatmap(numbers=True)

        With no columns named, every number column of the table is used (rows are the
        table's rows). Or name them: chart.heatmap(x="month", y="city", value="rain").
        numbers=True writes the number in each square. cluster=True puts similar rows
        and columns next to each other and draws a tree of how alike they are.
        """
        self._yes_no("heatmap", numbers=numbers, cluster=cluster)
        named = [r for r, v in (("x", x), ("y", y), ("value", value)) if v is not None]
        if named and len(named) < 3:
            raise PlotsimpleError(
                '.heatmap() needs all three of x=, y= and value=, for example\n'
                '    chart.heatmap(x="month", y="city", value="rain")\n'
                "or none of them, to use every number column of the table."
            )
        if named:
            check(self._data, x=x, y=y, value=value)
            _check_numbers(self._data, "heatmap", value=value)
        elif not any(_is_number_column(self._data[c]) for c in self._data.columns):
            raise PlotsimpleError(
                ".heatmap() found no number columns in your table.\n"
                'Name the columns to use: chart.heatmap(x="month", y="city", value="rain").'
            )
        self._need_axes("heatmap")
        return self._add("heatmap", "draw", _draw_heatmap, x=x, y=y, value=value,
                         numbers=numbers, cluster=cluster)

    @friendly
    def pie(self, labels=None, values=None):
        """A circle cut into slices, one slice per group. chart.pie(labels="fruit", values="count")

        Without values= each slice shows how many rows have that label.
        """
        if labels is None:
            raise PlotsimpleError('.pie() needs labels=, for example chart.pie(labels="fruit", values="count").')
        check(self._data, labels=labels, values=values)
        _check_numbers(self._data, "pie", values=values)
        self._need_axes("pie")
        return self._add("pie", "draw", _draw_pie, labels=labels, values=values)

    @friendly
    def polar(self, angle=None, distance=None, colour=None, degrees=True):
        """Dots on a round chart: how far round, and how far out. chart.polar(angle="direction", distance="speed")

        Angles are in degrees (a full turn is 360), going round from the right, the way
        maths measures angles. Use degrees=False if your angles are in radians.
        """
        self._yes_no("polar", degrees=degrees)
        if angle is None or distance is None:
            raise PlotsimpleError(
                '.polar() needs angle= and distance=, for example chart.polar(angle="direction", distance="speed").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, angle=angle, distance=distance, colour=group)
        _check_numbers(self._data, "polar", angle=angle, distance=distance)
        self._need_axes("polar", "polar")
        return self._add("polar", "draw", _draw_polar, angle=angle, distance=distance,
                         colour=colour, degrees=degrees)

    def _xyz(self, word, x, y, z, **more):
        if x is None or y is None or z is None:
            raise PlotsimpleError(
                f'.{word}() needs x=, y= and z=, for example chart.{word}(x="lon", y="lat", z="depth").'
            )
        check(self._data, x=x, y=y, z=z, **more)
        _check_numbers(self._data, word, x=x, y=y, z=z)

    @friendly
    def contour(self, x=None, y=None, z=None, filled=True):
        """A map where each colour band joins places with the same height. chart.contour(x="lon", y="lat", z="depth")

        z= is the height (or depth, or temperature). The points can be a full grid or
        scattered; scattered points are joined up with straight lines in between.
        filled=False draws only the lines, with their numbers written on them.
        """
        self._yes_no("contour", filled=filled)
        self._xyz("contour", x, y, z)
        self._need_axes("contour")
        return self._add("contour", "draw", _draw_contour, x=x, y=y, z=z, filled=filled)

    @friendly
    def surface(self, x=None, y=None, z=None):
        """A 3D landscape: z= is how high the ground is. chart.surface(x="lon", y="lat", z="depth")

        The points can be a full grid or scattered; scattered points are joined up.
        """
        self._xyz("surface", x, y, z)
        self._need_axes("surface", "3d")
        return self._add("surface", "draw", _draw_surface, x=x, y=y, z=z)

    @friendly
    def dots_3d(self, x=None, y=None, z=None, colour=None):
        """Dots floating in a 3D box. chart.dots_3d(x="a", y="b", z="c")

        colour="group" gives each group its own colour.
        """
        group, _ = colour_role(self._data, colour)
        self._xyz("dots_3d", x, y, z, colour=group)
        self._need_axes("dots_3d", "3d")
        return self._add("dots_3d", "draw", _draw_dots_3d, x=x, y=y, z=z, colour=colour)

    # standard names, same words
    bar = bars
    line = lines
    box = boxes
    violin = violins
