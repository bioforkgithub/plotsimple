"""Chart types: what you draw.

Calling two chart types on the same chart puts them on top of each other.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from .._core import PHASES
from .._data import frame_for, is_listlike
from .._errors import SimplydataError, column_error, friendly, suggest
from .._stats import SHOW, check_error, check_show, error_code, say_error
from ._helpers import check, colour_role, default_colour, palette_for, single_list

# all_dots: up to this many points in one group, dots are packed side by side (a swarm);
# above it they are spread at random (a strip), because a swarm would not fit.
SWARM_LIMIT = 100
JITTER_SEED = 0  # the same "random" spread every time the chart is drawn


def _remember(ctx, kind, table, names, **more):
    # row_order: how _frame() sorted the rows, so values given as a list (second_y) still
    # meet their own row
    more.setdefault("row_order", ctx.store.get("row_order"))
    ctx.last = dict(kind=kind, table=table, x=names.get("x"), y=names.get("y"),
                    colour=names.get("colour"), **more)


# ---------------------------------------------------------------- order of groups
# Groups (words) are drawn in the order they first appear in the table. To put them in
# another order, the rows are sorted (stably) so that the groups appear in that order: every
# chart type, colour group and statistics word then follows it, and no number changes.
# Month and weekday names are put in calendar order by themselves.
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_LONGER_SHORT = {"sept", "tues", "thur", "thurs"}


def _calendar_place(value):
    """(kind, place) of a month or weekday name: "Mar" -> ("month", 2); else None."""
    if not isinstance(value, str):
        return None
    word = value.strip().lower().rstrip(".")
    if len(word) < 3:
        return None
    for kind, names in (("month", _MONTHS), ("weekday", _WEEKDAYS)):
        for i, full in enumerate(names):
            if word == full or (full.startswith(word) and (len(word) == 3 or word in _LONGER_SHORT)):
                return kind, i
    return None


def calendar_order(values):
    """The values in calendar order if every one is a month name (or every one a weekday
    name), such as "Jan", "February", "mon"; otherwise None."""
    found = list(pd.unique(pd.Series(values).dropna()))
    if len(found) < 2:
        return None
    places = [_calendar_place(v) for v in found]
    if any(p is None for p in places) or len({p[0] for p in places}) != 1:
        return None
    return [v for _, v in sorted(zip([p[1] for p in places], found), key=lambda t: t[0])]


def _is_words(series) -> bool:
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_bool_dtype(series))


ORDER_WORDS = {"x": "order_x", "y": "order_y", "colour": "order_legend"}


def _numbers_cannot_be_ordered(word, col, where):
    return SimplydataError(
        f'.{word}() puts groups (words) in order, but "{col}" holds numbers or dates, '
        "which are always drawn in number order.\n"
        + (f"To turn the axis round, use chart.flip_{where}()." if where != "colour"
           else "A colour scale for numbers has no groups to put in order: leave out .order_legend().")
    )


def check_order_names(word, names, present, column):
    """Every name given to an order word must be in the column (friendly error if not)."""
    shown = [str(v) for v in present]
    for name in names:
        if str(name) not in shown:
            same = [s for s in shown if s.lower() == str(name).lower()]
            guess = same or suggest(str(name), shown, n=1)
            lines = [f'.{word}() got "{name}", but column "{column}" has no such value.']
            if guess:
                lines.append(f'Did you mean "{guess[0]}"?')
            lines.append("Values you have: " + ", ".join(shown))
            raise SimplydataError("\n".join(lines))


def _biggest_first_order(word, table, col, value, measure, biggest):
    if measure is None or (measure != "count" and value is None):
        raise SimplydataError(
            f".{word}(biggest_first=...) puts groups in order of their size, but this chart type has\n"
            "no size for a group. It works with bars, count_bars, lines, boxes, violins, all_dots\n"
            f'and heatmaps. Give the names in order instead: chart.{word}("Low", "Medium", "High").'
        )
    groups = table.groupby(col, sort=False, observed=True)
    sizes = groups.size() if measure == "count" else groups[value].agg(measure)
    sizes = sizes.dropna()
    return list(sizes.sort_values(ascending=not biggest, kind="stable").index)


def _apply_order(ctx, table, roles, measure=None):
    """Sort rows so the groups on the x-axis, the y-axis and in the legend appear in order.

    roles maps "x", "y", "colour" (and "value": the numbers biggest_first= sorts by) to the
    column names of this layer. Order: order_x/order_y/order_legend if given, else the
    order of a category column, else calendar order for month or weekday names.
    """
    keys, changed = [], {}
    asked_all = ctx.settings.get("order") or {}
    for where in ("x", "y", "colour"):
        col = roles.get(where)
        if col is None or col not in table.columns:
            continue
        asked, series = asked_all.get(where), table[col]
        is_category = isinstance(series.dtype, pd.CategoricalDtype)
        if asked is not None:
            word = ORDER_WORDS[where]
            if not (is_category or _is_words(series)):
                raise _numbers_cannot_be_ordered(word, col, where)
            if "names" in asked:
                known = list(pd.unique(series.dropna()))
                whole = ctx.store.get("whole_table")  # a panel may hold only some of the groups
                if whole is not None and col in whole.columns:
                    known += [v for v in pd.unique(whole[col].dropna()) if v not in set(known)]
                check_order_names(word, asked["names"], known, col)
                wanted = list(asked["names"])
            else:
                wanted = _biggest_first_order(word, table, col, roles.get("value"), measure,
                                              asked["biggest_first"])
        elif is_category:
            wanted = list(series.cat.categories)
        elif _is_words(series):
            wanted = calendar_order(series)
        else:
            wanted = None
        if not wanted:
            continue
        full = wanted + [v for v in pd.unique(series.dropna()) if v not in set(wanted)]
        place = {v: i for i, v in enumerate(full)}
        codes = series.astype(object).map(place)
        keys.append(pd.to_numeric(codes, errors="coerce").fillna(len(full)).to_numpy(dtype=float))
        if is_category:
            cats = list(series.cat.categories)
            changed[col] = [v for v in full if v in cats] + [v for v in cats if v not in place]
    if not keys:
        return table
    perm = np.lexsort(keys[::-1])  # the first key (x) matters most; the sort is stable
    table = table.iloc[perm].copy()
    for col, cats in changed.items():
        table[col] = table[col].cat.reorder_categories(cats)
    ctx.store["row_order"] = perm
    return table


_ORDER_EXAMPLE = {
    "x": 'chart.order_x("Low", "Medium", "High")',
    "y": 'chart.order_y("Low", "Medium", "High")',
    "colour": 'chart.order_legend("C", "B", "A")',
}


def _step_column(step, where):
    """The column a recorded chart step draws along the x-axis, the y-axis or as colours."""
    kw = step.kwargs
    if where == "colour":
        return kw.get("colour") if step.name != "pie" else kw.get("labels")
    if step.name in ("pie", "polar", "dots_3d", "surface", "contour"):
        return None
    if kw.get("sideways"):
        return kw.get("y" if where == "x" else "x")
    return kw.get(where)


def set_order(chart, where, names, biggest_first):
    """Shared by order_x, order_y and order_legend: check at once, then remember the order."""
    word, example = ORDER_WORDS[where], _ORDER_EXAMPLE[where]
    if len(names) == 1 and is_listlike(names[0]):
        names = tuple(names[0])
    if names and biggest_first is not None:
        raise SimplydataError(f".{word}() takes names or biggest_first=, not both.\nFor example {example}")
    if not names and biggest_first is None:
        raise SimplydataError(
            f".{word}() needs the names in the order you want them.\nFor example {example}"
            + ("" if where == "colour" else f", or chart.{word}(biggest_first=True)"))
    if biggest_first is not None and not isinstance(biggest_first, bool):
        raise SimplydataError(f"biggest_first= in .{word}() should be True or False, not {biggest_first!r}.")
    for name in names:
        if not isinstance(name, (str, int, float, np.integer, np.floating)) or isinstance(name, bool):
            raise SimplydataError(f".{word}() needs names (words), not {name!r}.\nFor example {example}")
    repeated = [n for i, n in enumerate(names) if n in names[:i]]
    if repeated:
        raise SimplydataError(f'.{word}() got "{repeated[0]}" twice. Give each name once.')
    # check against the chart types already added (others are checked when drawn)
    data = chart._data
    for step in chart._steps:
        if step.phase != PHASES["draw"]:
            continue
        col = _step_column(step, where)
        if not isinstance(col, str) or col not in data.columns:
            continue
        if not (_is_words(data[col]) or isinstance(data[col].dtype, pd.CategoricalDtype)):
            raise _numbers_cannot_be_ordered(word, col, where)
        if names:
            check_order_names(word, names, pd.unique(data[col].dropna()), col)
    order = dict(chart._settings.get("order") or {})
    order[where] = {"names": list(names)} if names else {"biggest_first": biggest_first}
    return chart._set(order=order)


def group_values(chart, column, words="appearance"):
    """The groups of a column in the order panels (one_chart_per) and animation frames use.

    The names given to order_x / order_y / order_legend, when that axis or legend shows this
    column; else the order of a category column; else calendar order for month or weekday
    names; numbers and dates go from small to big. Other words keep the order they appear in
    the table (words="appearance") or go A to Z (words="sorted").
    """
    data = chart._data
    series = data[column]
    present = list(pd.unique(series.dropna()))
    asked_all = chart._settings.get("order") or {}
    wanted = None
    for where, asked in asked_all.items():
        if "names" not in asked:
            continue
        if any(_step_column(step, where) == column for step in chart._steps if step.phase == PHASES["draw"]):
            wanted = list(asked["names"])
            break
    if wanted is None and isinstance(series.dtype, pd.CategoricalDtype):
        wanted = list(series.cat.categories)
    if wanted is None and _is_words(series):
        wanted = calendar_order(series)
    if wanted is None and (not _is_words(series) or words == "sorted"):
        try:
            return sorted(present)
        except TypeError:  # a mix that cannot be sorted: keep the order in the table
            return present
    if wanted is None:
        return present
    shown = [v for v in wanted if v in set(present)]
    return shown + [v for v in present if v not in set(shown)]


def _hue(table, n):
    """hue= for seaborn, with the groups in the order they appear (seaborn may sort them)."""
    col = n.get("colour")
    kw = {"hue": col}
    if col is not None and (_is_words(table[col]) or isinstance(table[col].dtype, pd.CategoricalDtype)):
        kw["hue_order"] = list(pd.unique(table[col].dropna()))
    return kw


def _number_role(data, x, y):
    """Which of x and y holds the numbers of a boxes/violins/all_dots chart (for biggest_first=)."""
    for role, col in (("y", y), ("x", x)):
        if is_listlike(col):
            values = pd.Series(list(col))
        elif isinstance(col, str) and col in data.columns:
            values = data[col]
        else:
            continue
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            return role
    return None


def _frame(ctx, on_x="x", on_y="y", measure=None, size_role=None, **roles):
    """frame_for(), then the groups put in order (order_x, order_y, order_legend, calendar).

    on_x / on_y: the role drawn along each axis (sideways bars swap them).
    measure / size_role: what biggest_first= sorts the groups by ("mean" of the y role ...).
    """
    ctx.store.pop("row_order", None)
    table, n = frame_for(ctx.data, **roles)
    place = {"x": n.get(on_x) if on_x else None, "y": n.get(on_y) if on_y else None,
             "colour": n.get("colour"), "value": n.get(size_role) if size_role else None}
    return _apply_order(ctx, table, place, measure), n


# ---------------------------------------------------------------- where the empty cells are
EMPTY_COLOUR = "#D55E00"  # Okabe-Ito vermilion: stands out from the grey of filled cells
FILLED_COLOUR = "#E3E3E3"


def _empty_cell_grid(data, columns):
    """True for every empty cell: missing, NaN, or text that is only spaces (as in the table words)."""
    part = data[columns]
    empty = part.isna()
    for col in columns:
        if part[col].dtype == object or pd.api.types.is_string_dtype(part[col]):
            empty[col] = empty[col] | part[col].map(lambda v: isinstance(v, str) and not v.strip())
    return empty


def _draw_empty_cells(ctx, columns):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    data, ax = ctx.data, ctx.ax
    columns = list(columns) if columns else list(data.columns)
    empty = _empty_cell_grid(data, columns)
    n_rows = len(data)
    marked = ctx.settings.get("colour") or EMPTY_COLOUR
    ax.imshow(empty.to_numpy(dtype=float).reshape(n_rows, len(columns)), aspect="auto", interpolation="nearest",
              cmap=ListedColormap([FILLED_COLOUR, marked]), vmin=0, vmax=1,
              extent=(-0.5, len(columns) - 0.5, n_rows + 0.5, 0.5))
    counts = empty.sum().astype(int)
    ax.set_xticks(range(len(columns)), labels=[f"{c}\n{counts[c]} empty" for c in columns])
    ax.tick_params(axis="x", length=0)
    ax.grid(False)
    for i in range(1, len(columns)):  # a thin gap between the columns
        ax.axvline(i - 0.5, color=matplotlib.rcParams["axes.facecolor"], linewidth=3)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("row (in table order)")
    total = int(counts.sum())
    ax.legend(handles=[Patch(color=marked, label=f"empty ({total} cells)"), Patch(color=FILLED_COLOUR, label="filled")],
              loc="upper left", bbox_to_anchor=(1.01, 1))


# ---------------------------------------------------------------- bars
def _draw_bars(ctx, x, y, colour, show, error, stacked, sideways):
    group, fixed = colour_role(ctx.data, colour)
    how = SHOW[show]
    table, n = _frame(ctx, on_x="y" if sideways else "x", on_y="x" if sideways else "y",
                      measure=how, size_role="y", x=x, y=y, colour=group)
    ax = ctx.ax

    if stacked:
        cols = [n["x"], n["colour"]]
        if how == "count":
            wide = table.groupby(cols, sort=False).size().unstack(n["colour"])
        else:
            wide = table.groupby(cols, sort=False)[n["y"]].agg(how).unstack(n["colour"])
        # unstack sorts by name; keep the order the groups appear in (order_x, order_legend)
        wide = wide.reindex(index=[v for v in pd.unique(table[n["x"]].dropna()) if v in wide.index],
                            columns=[v for v in pd.unique(table[n["colour"]].dropna()) if v in wide.columns])
        if sideways:
            wide = wide.iloc[::-1]  # barh draws the first row at the bottom; put it at the top
        wide.plot(kind="barh" if sideways else "bar", stacked=True, ax=ax, rot=0, width=0.8,
                  color=palette_for(table, n["colour"], ctx.settings))
        ax.legend(title=n["colour"])
        if sideways:
            ax.set_xlabel(n["y"] or "count")
        else:
            ax.set_ylabel(n["y"] or "count")
    else:
        kw = dict(data=table, **_hue(table, n), ax=ax, saturation=1)  # exact colours, as asked
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
                kw["capsize"] = 0.1
                kw["err_kws"] = {"linewidth": 1.3}
            sns.barplot(**kw)
    _remember(ctx, "bars", table, n, sideways=sideways)


# ---------------------------------------------------------------- lines
def _draw_lines(ctx, x, y, colour, show, error, dots, dashed):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, measure=SHOW[show], size_role="y", x=x, y=y, colour=group)
    kw = dict(data=table, x=n["x"], y=n["y"], **_hue(table, n), ax=ctx.ax,
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
    _remember(ctx, "lines", table, n, show=show)


# ---------------------------------------------------------------- dots
def _draw_dots(ctx, x, y, colour, size, shape):
    group, fixed = colour_role(ctx.data, colour)
    size_col = size if not isinstance(size, (int, float)) else None
    table, n = _frame(ctx, x=x, y=y, colour=group, size=size_col, shape=shape)
    kw = dict(data=table, x=n["x"], y=n["y"], **_hue(table, n), size=n["size"], style=n["shape"], ax=ctx.ax)
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
            raise SimplydataError(
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
        raise SimplydataError(f".{word}() needs at least 2 different x values and 2 different y values.")
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
        raise SimplydataError(
            f".{word}() could not join your points into a surface.\n"
            "It needs at least 3 points that do not all lie on one straight line."
        ) from None
    return gx, gy, gz


# ---------------------------------------------------------------- area
def _draw_area(ctx, x, y, colour, stacked):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, measure="mean", size_role="y", x=x, y=y, colour=group)
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
    table, n = _frame(ctx, x=x, colour=group)
    kw = dict(data=table, x=n["x"], **_hue(table, n), bins=bins, ax=ctx.ax)
    sns.histplot(**_seaborn_colours(kw, table, n, ctx, fixed))
    ctx.ax.set_ylabel("count")
    _remember(ctx, "histogram", table, n)


# ---------------------------------------------------------------- boxes and violins
def _draw_boxes(ctx, x, y, colour, violins):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, measure="median", size_role=_number_role(ctx.data, x, y), x=x, y=y, colour=group)
    kw = dict(data=table, x=n["x"], y=n["y"], **_hue(table, n), ax=ctx.ax, saturation=1)
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


# ---------------------------------------------------------------- all dots
def _draw_all_dots(ctx, x, y, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, measure="median", size_role=_number_role(ctx.data, x, y), x=x, y=y, colour=group)
    dodge = bool(n["colour"]) and n["colour"] not in (n["x"], n["y"])
    kw = dict(data=table, x=n["x"], y=n["y"], **_hue(table, n), ax=ctx.ax, dodge=dodge)
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
    _remember(ctx, "all_dots", table, n)


# ---------------------------------------------------------------- density
def _draw_density(ctx, x, colour):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, x=x, colour=group)
    # common_norm=False: each group's curve has area 1, so small groups are not squashed
    kw = dict(data=table, x=n["x"], **_hue(table, n), ax=ctx.ax, common_norm=False)
    sns.kdeplot(**_seaborn_colours(kw, table, n, ctx, fixed))
    _remember(ctx, "density", table, n)


# ---------------------------------------------------------------- heatmap
def _heatmap_table(ctx, x, y, value):
    data = ctx.data
    if value is not None:
        table, n = _frame(ctx, measure="mean", size_role="value", x=x, y=y, value=value)
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
    wide, label = _heatmap_table(ctx, x, y, value)
    if wide.empty:
        raise SimplydataError(".heatmap() found no numbers to draw.")
    ax = ctx.ax
    trees = None
    if cluster:
        if wide.isna().any().any():
            raise SimplydataError(
                "cluster=True cannot sort rows and columns that have empty cells.\n"
                "Fill or remove the empty cells first, or use cluster=False."
            )
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            raise SimplydataError("cluster=True needs at least 2 rows and 2 columns.")
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
    table, n = _frame(ctx, on_x=None, on_y=None, colour=labels, labels=labels, values=values)
    if n["values"] is None:
        sizes = table[n["labels"]].value_counts(sort=False, dropna=True)
        sizes = sizes.reindex(pd.unique(table[n["labels"]].dropna()))
    else:
        sizes = table.groupby(n["labels"], sort=False)[n["values"]].sum()
    if (sizes < 0).any():
        raise SimplydataError(".pie() cannot show values below 0. A slice cannot be smaller than nothing.")
    if sizes.sum() == 0:
        raise SimplydataError(".pie() needs at least one value bigger than 0.")
    ax = ctx.ax
    ax.pie(sizes.to_numpy(dtype=float), labels=[str(i) for i in sizes.index],
           colors=palette_for(table, n["labels"], ctx.settings), autopct="%1.0f%%",
           startangle=90, counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.grid(False)
    ctx.store["pie_sizes"] = sizes
    ctx.last = dict(kind="pie", table=table, x=n["labels"], y=n["values"], colour=None)


# ---------------------------------------------------------------- band
BAND_SHADE = 0.25  # how solid a band is: 0 is invisible, 1 is solid


def _rows_low_above_high(low, high):
    lo = pd.to_numeric(pd.Series(list(low)), errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(pd.Series(list(high)), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return int(np.sum(lo > hi))


def _band_swapped_error(count, low, high):
    low = low if isinstance(low, str) else "the first list"
    high = high if isinstance(high, str) else "the second list"
    return SimplydataError(
        f".band() needs the low edge first: in {count} row{'s' if count != 1 else ''} "
        f'"{low}" is bigger than "{high}".\n'
        f"Did you swap them? Write between=[{high!r}, {low!r}] if so."
    )


def _draw_band(ctx, x, low, high, colour, label):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, on_y=None, x=x, low=low, high=high, colour=group)
    swapped = _rows_low_above_high(table[n["low"]], table[n["high"]])
    if swapped:
        raise _band_swapped_error(swapped, low, high)
    ax = ctx.ax
    entries = []
    for (name, rows), c in zip(_groups(table, n["colour"]), _group_colours(table, n["colour"], ctx, fixed)):
        rows = rows[rows[n["x"]].notna()]
        repeated = rows[n["x"]][rows[n["x"]].duplicated()]
        if len(repeated):
            where = f' in group "{name}"' if name is not None else ""
            raise SimplydataError(
                f'.band() found more than one row with {n["x"]} = {repeated.iloc[0]}{where}.\n'
                "A band needs one low and one high value at each x"
                + (". Add colour= to draw one band per group." if name is None else ".")
            )
        if _ordered(rows[n["x"]]):
            rows = rows.sort_values(n["x"], kind="stable")
        # empty cells leave a gap in the band rather than a made-up edge
        shape = ax.fill_between(rows[n["x"]].to_numpy(),
                                rows[n["low"]].to_numpy(dtype=float),
                                rows[n["high"]].to_numpy(dtype=float),
                                color=c, alpha=BAND_SHADE, linewidth=0)
        text = label if name is None else str(name)
        if text:
            entries.append((shape, text))
    if ax.get_xlabel() == "":
        ax.set_xlabel(n["x"])
    store = ctx.store.setdefault("band_entries", [])
    store.extend((h, t, n["colour"]) for h, t in entries)
    if ctx.last is None:  # a band alone is the chart; on top of lines it leaves them in charge
        _remember(ctx, "band", table, {"x": n["x"], "y": n["high"], "colour": n["colour"]})


def _band_legend(ctx):
    """Add the bands to the legend (after every chart type, so lines cannot replace it)."""
    entries = ctx.store.pop("band_entries", [])
    if not entries:
        return
    ax = ctx.ax
    old = ax.get_legend()
    handles, labels, title = [], [], None
    if old is not None:
        handles = list(old.legend_handles)
        labels = [t.get_text() for t in old.get_texts()]
        title = old.get_title().get_text() or None
        old.remove()
    for handle, text, group in entries:
        if text not in labels:  # a group already in the legend (its line) keeps one entry
            handles.append(handle)
            labels.append(text)
        if title is None and group:
            title = group
    ax.legend(handles, labels, title=title)


# ---------------------------------------------------------------- polar
def _draw_polar(ctx, angle, distance, colour, degrees):
    group, fixed = colour_role(ctx.data, colour)
    table, n = _frame(ctx, on_x=None, on_y=None, angle=angle, distance=distance, colour=group)
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
    table, n = _frame(ctx, on_x=None, on_y=None, x=x, y=y, z=z, colour=group)
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
            raise SimplydataError(
                f".{word}() draws {_KINDS[kind]}, but this chart already has {others}, "
                f"which make{'s' if len(drawn) == 1 else ''} {_KINDS[current]}.\n"
                f"They cannot share one chart. Start a new chart for .{word}(), for example:\n"
                f"    other = sdata.chart(table)\n"
                f"    other.{word}(...)"
            )
        self._set(projection=kind)

    @friendly
    def bars(self, x=None, y=None, colour=None, show="average", error=None,
             stacked=False, sideways=False):
        """Bars. chart.bars(x="name", y="marks") (with sdata.example("marks"))

        colour="class" gives each group its own colour, or colour="blue" one colour.
        show= says what each bar stands for when a group has many rows: "average" (default),
        "median", "total", "count", "smallest", "biggest" or "standard deviation".
        error="standard deviation", "standard error" or "confidence interval" adds error
        bars (with show="average"); a line says which one was drawn.
        stacked=True puts groups on top of each other; sideways=True lays bars down.
        """
        show = check_show("bars", show)
        check_error("bars", error, show)
        error = error_code("bars", error)
        x, y = single_list(x, y)
        if x is None:
            raise SimplydataError('.bars() needs x=, for example chart.bars(x="name", y="marks").')
        if y is None and SHOW[show] != "count":
            raise SimplydataError(
                '.bars() needs y=, for example chart.bars(x="name", y="marks").\n'
                'To count rows instead, use chart.bars(x="name", show="count").'
            )
        group, _ = colour_role(self._data, colour)
        if stacked and group is None:
            raise SimplydataError('stacked=True needs groups to stack: add colour="a column".')
        if stacked and error:
            raise SimplydataError("Error bars cannot be drawn on stacked bars. Remove error= or stacked=True.")
        check(self._data, x=x, y=y, colour=group)
        self._need_axes("bars")
        if error:
            say_error(error)
        return self._add("bars", "draw", _draw_bars, x=x, y=y, colour=colour, show=show,
                         error=error, stacked=stacked, sideways=sideways)

    @friendly
    def lines(self, x=None, y=None, colour=None, show="average", error=None, dots=False, dashed=False):
        """Lines. chart.lines(x="month", y="temperature", colour="city") (with sdata.example("weather"))

        colour="city" draws one line per group. dots=True marks each point.
        dashed=True draws a dashed line. show= works as in bars (not "count").
        error="standard deviation", "standard error" or "confidence interval" adds a shaded
        band; a line says which one was drawn.
        """
        show = check_show("lines", show)
        if SHOW[show] == "count":
            raise SimplydataError('.lines() cannot use show="count". Use "average", "median" or "total".')
        check_error("lines", error, show)
        error = error_code("lines", error)
        x, y = single_list(x, y)
        if x is None or y is None:
            raise SimplydataError('.lines() needs x= and y=, for example chart.lines(x="month", y="temperature").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        self._need_axes("lines")
        if error:
            say_error(error, "Shaded band")
        return self._add("lines", "draw", _draw_lines, x=x, y=y, colour=colour, show=show,
                         error=error, dots=dots, dashed=dashed)

    @friendly
    def dots(self, x=None, y=None, colour=None, size=None, shape=None):
        """Dots (a scatter chart). chart.dots(x="hours", y="marks") (with sdata.example("marks"))

        colour="class" colours dots by group. size="height" makes dots bigger for bigger
        values, or size=10 makes all dots that big. shape="subject" gives each group a shape.
        """
        x, y = single_list(x, y)
        if x is None or y is None:
            raise SimplydataError('.dots() needs x= and y=, for example chart.dots(x="hours", y="marks").')
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
                raise SimplydataError(f"{option}= in .{word}() should be True or False, not {value!r}.")

    @friendly
    def area(self, x=None, y=None, colour=None, stacked=False):
        """A line with the space under it coloured in.
        chart.area(x="year", y="population", colour="country") (with sdata.example("growth"))

        colour="country" gives each group its own area; stacked=True piles the groups on top
        of each other, so the top edge shows the total.
        """
        self._yes_no("area", stacked=stacked)
        x, y = single_list(x, y)
        if x is None or y is None:
            raise SimplydataError('.area() needs x= and y=, for example chart.area(x="year", y="population").')
        group, _ = colour_role(self._data, colour)
        if stacked and group is None:
            raise SimplydataError('stacked=True needs groups to stack: add colour="a column".')
        check(self._data, x=x, y=y, colour=group)
        _check_numbers(self._data, "area", y=y)
        self._need_axes("area")
        return self._add("area", "draw", _draw_area, x=x, y=y, colour=colour, stacked=stacked)

    @friendly
    def histogram(self, x=None, bins=10, colour=None):
        """Bars that count how many values fall in each range. chart.histogram(x="marks", bins=10)
        (with sdata.example("marks"))

        bins= is how many bars (or a list of where each bar starts and ends).
        colour="class" draws one see-through histogram per group.
        """
        if x is None:
            raise SimplydataError('.histogram() needs x=, for example chart.histogram(x="marks").')
        if isinstance(bins, bool) or not (
            (isinstance(bins, (int, np.integer)) and bins >= 1)
            or (isinstance(bins, (list, tuple, np.ndarray)) and len(bins) >= 2
                and all(isinstance(b, (int, float, np.number)) for b in bins)
                and all(a < b for a, b in zip(bins[:-1], bins[1:])))
        ):
            raise SimplydataError(
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
            raise SimplydataError(
                f'.{word}() needs y= (and usually x=), for example chart.{word}(x="class", y="marks").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        self._need_axes(word)
        return self._add(word, "draw", _draw_boxes, x=x, y=y, colour=colour, violins=word == "violins")

    @friendly
    def boxes(self, x=None, y=None, colour=None):
        """Boxes that show the middle half of the values and the median.
        chart.boxes(x="class", y="marks") (with sdata.example("marks"))

        The line in the box is the median, the box holds the middle half of the values,
        the whiskers reach the rest, and lone dots are values far away from the others.
        """
        return self._boxes_or_violins("boxes", x, y, colour)

    @friendly
    def violins(self, x=None, y=None, colour=None):
        """Shapes that are wide where many values are. chart.violins(x="class", y="marks")
        (with sdata.example("marks"))

        Each violin stops at the smallest and the largest value; the small box inside
        shows the middle half of the values.
        """
        return self._boxes_or_violins("violins", x, y, colour)

    @friendly
    def count_bars(self, x=None, colour=None, sideways=False):
        """Bars that count how many rows each group has. chart.count_bars(x="fruit", colour="shop")
        (with sdata.example("fruit"))

        colour="shop" splits each bar by group; sideways=True lays the bars down.
        """
        self._yes_no("count_bars", sideways=sideways)
        if x is None:
            raise SimplydataError('.count_bars() needs x=, for example chart.count_bars(x="fruit").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, colour=group)
        self._need_axes("count_bars")
        return self._add("count_bars", "draw", _draw_bars, x=x, y=None, colour=colour, show="count",
                         error=None, stacked=False, sideways=sideways)

    @friendly
    def all_dots(self, x=None, y=None, colour=None):
        """Every value as a dot, spread out so no dot hides another.
        chart.all_dots(x="class", y="marks") (with sdata.example("marks"))

        Good on top of .boxes() or .violins(), to show each value as well.
        """
        if x is None and y is None:
            raise SimplydataError(
                '.all_dots() needs y= (and usually x=), for example chart.all_dots(x="class", y="marks").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, y=y, colour=group)
        self._need_axes("all_dots")
        return self._add("all_dots", "draw", _draw_all_dots, x=x, y=y, colour=colour)

    @friendly
    def density(self, x=None, colour=None):
        """A smooth curve that is high where many values are. chart.density(x="marks", colour="class")
        (with sdata.example("marks"))

        colour="class" draws one curve per group; each curve has an area of 1.
        """
        if x is None:
            raise SimplydataError('.density() needs x=, for example chart.density(x="marks").')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, colour=group)
        _check_numbers(self._data, "density", x=x)
        self._need_axes("density")
        return self._add("density", "draw", _draw_density, x=x, colour=colour)

    @friendly
    def heatmap(self, x=None, y=None, value=None, numbers=False, cluster=False):
        """A grid of coloured squares, one colour for each number.
        chart.heatmap(x="month", y="city", value="rain", numbers=True) (with sdata.example("weather"))

        With no columns named, every number column of the table is used (rows are the
        table's rows): chart.heatmap(numbers=True).
        numbers=True writes the number in each square. cluster=True puts similar rows
        and columns next to each other and draws a tree of how alike they are.
        """
        self._yes_no("heatmap", numbers=numbers, cluster=cluster)
        named = [r for r, v in (("x", x), ("y", y), ("value", value)) if v is not None]
        if named and len(named) < 3:
            raise SimplydataError(
                '.heatmap() needs all three of x=, y= and value=, for example\n'
                '    chart.heatmap(x="month", y="city", value="rain")\n'
                "or none of them, to use every number column of the table."
            )
        if named:
            check(self._data, x=x, y=y, value=value)
            _check_numbers(self._data, "heatmap", value=value)
        elif not any(_is_number_column(self._data[c]) for c in self._data.columns):
            raise SimplydataError(
                ".heatmap() found no number columns in your table.\n"
                'Name the columns to use: chart.heatmap(x="month", y="city", value="rain").'
            )
        self._need_axes("heatmap")
        return self._add("heatmap", "draw", _draw_heatmap, x=x, y=y, value=value,
                         numbers=numbers, cluster=cluster)

    @friendly
    def empty_cells(self, columns=None):
        """A map of the empty cells: one row of the chart for each row of the table, one strip
        for each column, with empty cells coloured. chart.empty_cells() (with sdata.example("shop"))

        Under each column it says how many of its cells are empty. columns=["sales", "customers"]
        shows only those columns. A cell is empty when it is missing or holds only spaces.
        """
        data = self._data
        if data is None or len(data.columns) == 0:
            raise SimplydataError(
                '.empty_cells() needs a table, for example sdata.chart(sdata.example("shop")).empty_cells().')
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            if not is_listlike(columns) or not list(columns):
                raise SimplydataError('columns= should be a list of column names, like columns=["sales", "customers"].')
            columns = list(columns)
            for name in columns:
                if name not in data.columns:
                    raise column_error(name, data.columns)
        self._need_axes("empty_cells")
        return self._add("empty_cells", "draw", _draw_empty_cells, columns=columns)

    @friendly
    def pie(self, labels=None, values=None):
        """A circle cut into slices, one slice per group. chart.pie(labels="fruit", values="count")
        (with sdata.example("fruit"))

        Without values= each slice shows how many rows have that label.
        """
        if labels is None:
            raise SimplydataError('.pie() needs labels=, for example chart.pie(labels="fruit", values="count").')
        check(self._data, labels=labels, values=values)
        _check_numbers(self._data, "pie", values=values)
        self._need_axes("pie")
        return self._add("pie", "draw", _draw_pie, labels=labels, values=values)

    @friendly
    def polar(self, angle=None, distance=None, colour=None, degrees=True):
        """Dots on a round chart: how far round, and how far out.
        chart.polar(angle="wind_direction", distance="wind_speed", colour="city") (with sdata.example("weather"))

        Angles are in degrees (a full turn is 360), going round from the right, the way
        maths measures angles. Use degrees=False if your angles are in radians.
        """
        self._yes_no("polar", degrees=degrees)
        if angle is None or distance is None:
            raise SimplydataError(
                '.polar() needs angle= and distance=, for example chart.polar(angle="wind_direction", distance="wind_speed").'
            )
        group, _ = colour_role(self._data, colour)
        check(self._data, angle=angle, distance=distance, colour=group)
        _check_numbers(self._data, "polar", angle=angle, distance=distance)
        self._need_axes("polar", "polar")
        return self._add("polar", "draw", _draw_polar, angle=angle, distance=distance,
                         colour=colour, degrees=degrees)

    def _xyz(self, word, x, y, z, **more):
        if x is None or y is None or z is None:
            raise SimplydataError(
                f'.{word}() needs x=, y= and z=, for example chart.{word}(x="east", y="north", z="height").'
            )
        check(self._data, x=x, y=y, z=z, **more)
        _check_numbers(self._data, word, x=x, y=y, z=z)

    @friendly
    def contour(self, x=None, y=None, z=None, filled=True):
        """A map where each colour band joins places with the same height.
        chart.contour(x="east", y="north", z="height") (with sdata.example("island"))

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
        """A 3D landscape: z= is how high the ground is.
        chart.surface(x="east", y="north", z="height") (with sdata.example("island"))

        The points can be a full grid or scattered; scattered points are joined up.
        """
        self._xyz("surface", x, y, z)
        self._need_axes("surface", "3d")
        return self._add("surface", "draw", _draw_surface, x=x, y=y, z=z)

    @friendly
    def dots_3d(self, x=None, y=None, z=None, colour=None):
        """Dots floating in a 3D box. chart.dots_3d(x="hours", y="height", z="marks", colour="class")
        (with sdata.example("marks"))

        colour="class" gives each group its own colour.
        """
        group, _ = colour_role(self._data, colour)
        self._xyz("dots_3d", x, y, z, colour=group)
        self._need_axes("dots_3d", "3d")
        return self._add("dots_3d", "draw", _draw_dots_3d, x=x, y=y, z=z, colour=colour)

    @friendly
    def band(self, x=None, between=None, colour=None, label=None):
        """A shaded band between a low and a high column, along x.
        chart.band(x="month", between=["lowest", "highest"]) (with a table that has those columns)

        between=["low column", "high column"] gives the bottom and the top edge at each x:
        a range of temperatures, a confidence band you worked out, or a smallest-to-biggest
        band around a line. Draw .lines() as well to show the middle.
        colour="city" draws one band per group (in the group's colour), or colour="blue" one
        colour. label="Range" names the band in the legend. Empty cells leave a gap.
        """
        example = 'chart.band(x="month", between=["lowest", "highest"])'
        if x is None or between is None:
            raise SimplydataError(f".band() needs x= and between=, for example {example}.")
        if isinstance(between, str) or not is_listlike(between) or len(between) != 2:
            raise SimplydataError(
                f".band() needs two columns in between=: the low edge and the high edge.\nFor example {example}.")
        low, high = between
        if isinstance(low, str) and low == high:
            raise SimplydataError(f'.band() needs two different columns in between=, not "{low}" twice.')
        if label is not None and not isinstance(label, str):
            raise SimplydataError(f'label= should be words, for example label="Range", not {label!r}.')
        group, _ = colour_role(self._data, colour)
        check(self._data, x=x, low=low, high=high, colour=group)
        for edge in (low, high):
            if isinstance(edge, str) and not _is_number_column(self._data[edge]):
                raise SimplydataError(
                    f'.band() needs numbers for its edges, but column "{edge}" holds words or other things.\n'
                    "Pick two columns with numbers in them.")
        if isinstance(low, str) and isinstance(high, str):
            swapped = _rows_low_above_high(self._data[low], self._data[high])
            if swapped:
                raise _band_swapped_error(swapped, low, high)
        self._need_axes("band")
        self._add("band", "draw", _draw_band, x=x, low=low, high=high, colour=colour, label=label)
        return self._replace("band_legend", "extra", _band_legend)

    # standard names, same words
    bar = bars
    line = lines
    box = boxes
    violin = violins
