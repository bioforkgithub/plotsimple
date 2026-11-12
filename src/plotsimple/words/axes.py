"""Axes: ranges, log scales, ticks, second y-axis."""

from __future__ import annotations

import datetime as _dt
import numbers

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like, to_hex

from .._data import is_listlike
from .._errors import PlotsimpleError, column_error, friendly
from .._style import palette
from ._helpers import check

SECOND_KINDS = ("lines", "bars", "dots")


# ---------------------------------------------------------------- shared helpers
def _is_value(v) -> bool:
    """A number or a date: something an axis can be set to."""
    if isinstance(v, bool):
        return False
    return isinstance(v, (numbers.Real, _dt.date, np.datetime64, pd.Timestamp))


def _check_value(step, name, v, example):
    if v is not None and not _is_value(v):
        raise PlotsimpleError(f".{step}() needs numbers, but {name} is {v!r}.\nFor example: {example}")


def tick_position(ax, axis, value):
    """Where a value sits on an axis.

    Numbers stay numbers. A word such as "Ben" is looked up among the tick labels,
    because bar charts put their groups at 0, 1, 2 ... and write the names under them.
    """
    if not isinstance(value, str):
        return value
    get_ticks = ax.get_xticks if axis == "x" else ax.get_yticks
    get_labels = ax.get_xticklabels if axis == "x" else ax.get_yticklabels
    names = [t.get_text() for t in get_labels()]
    if value in names:
        return get_ticks()[names.index(value)]
    shown = [n for n in names if n]
    raise PlotsimpleError(
        f'"{value}" is not one of the names on the {axis}-axis.\n'
        + ("Names you have: " + ", ".join(shown) if shown else f"Use a number for {axis}= instead.")
    )


# ---------------------------------------------------------------- ranges
def _range(ctx, axis, low, high):
    ax = ctx.ax
    get_lim = ax.get_xlim if axis == "x" else ax.get_ylim
    set_lim = ax.set_xlim if axis == "x" else ax.set_ylim
    inverted = ax.xaxis_inverted() if axis == "x" else ax.yaxis_inverted()
    conv = ax.xaxis.convert_units if axis == "x" else ax.yaxis.convert_units
    cur_low, cur_high = sorted(get_lim())
    new_low = cur_low if low is None else conv(low)
    new_high = cur_high if high is None else conv(high)
    if new_low >= new_high:
        raise PlotsimpleError(
            f".range_{axis}(): the chart's {'top' if low is not None else 'bottom'} end is "
            f"{(new_high if low is not None else new_low):g}, so the range would be empty.\n"
            f"Give both ends, for example chart.range_{axis}(0, 100)."
        )
    if inverted:
        set_lim(new_high, new_low)
    else:
        set_lim(new_low, new_high)


# ---------------------------------------------------------------- log scales
def _set_log(ctx, axis):
    (ctx.ax.set_xscale if axis == "x" else ctx.ax.set_yscale)("log")


def _values_on(ctx, axis):
    """The values that sit on this axis, with a name for the message."""
    ax, last = ctx.ax, ctx.last
    bars = [c for c in ax.containers if isinstance(c, mpl.container.BarContainer)]
    if bars:
        value_axis = "x" if getattr(bars[0], "orientation", "vertical") == "horizontal" else "y"
        if axis == value_axis:
            vals = [v for c in bars for v in np.ravel(c.datavalues)]
            name = (last or {}).get("y") or "the bars"
            return np.asarray(vals, dtype=float), name
    if last is None:
        return None, None
    col = last.get(axis)
    if last.get("sideways"):
        col = last.get("y" if axis == "x" else "x")
    table = last.get("table")
    if col is None or table is None or col not in table.columns:
        return None, None
    values = table[col]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise PlotsimpleError(
            f'.log_scale_{axis}() needs numbers, but "{col}" on the {axis}-axis holds words or groups.\n'
            "Use the log scale on the axis that shows numbers."
        )
    return values.to_numpy(dtype=float), col


def _check_log(ctx, axis):
    values, name = _values_on(ctx, axis)
    if values is None:
        return
    values = values[~np.isnan(values)]
    bad = values[values <= 0]
    if len(bad):
        raise PlotsimpleError(
            f'.log_scale_{axis}() cannot show zero or negative numbers, and "{name}" has {len(bad)} '
            f"of them (smallest: {bad.min():g}).\n"
            "A log scale only works when every value is bigger than 0. You can:\n"
            "  * remove those rows from your table first, or\n"
            f"  * leave out .log_scale_{axis}() and use .range_{axis}(...) to zoom in instead."
        )


# ---------------------------------------------------------------- ticks
def _ticks(ctx, axis, values, labels):
    ax = ctx.ax
    names = [v if isinstance(v, str) else None for v in values]
    where = [tick_position(ax, axis, v) for v in values]
    if labels is None and any(names):
        # keep the names under the bars, not the numbers 0, 1, 2 behind them
        labels = [n if n is not None else f"{v:g}" for n, v in zip(names, where)]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(where, labels=labels)


# ---------------------------------------------------------------- rotate and flip
def _rotate(ctx, axis, angle):
    ax = ctx.ax
    turned = angle % 180 != 0
    if axis == "x":
        ha = "center" if not turned else ("right" if 0 < angle % 360 < 180 else "left")
        va = "center" if abs(angle) == 90 else "top"
        labels = ax.get_xticklabels()
    else:
        ha, va = "right", "center"
        labels = ax.get_yticklabels()
    for label in labels:
        label.set_rotation(angle)
        label.set_horizontalalignment(ha)
        label.set_verticalalignment(va)
        label.set_rotation_mode("anchor" if turned else "default")


def _flip(ctx, axis):
    ax = ctx.ax
    if axis == "x" and not ax.xaxis_inverted():
        ax.invert_xaxis()
    elif axis == "y" and not ax.yaxis_inverted():
        ax.invert_yaxis()


# ---------------------------------------------------------------- second y-axis
def _used_colours(ax):
    used = set()
    for line in ax.lines:
        used.add(to_hex(line.get_color()))
    for patch in ax.patches:
        used.add(to_hex(patch.get_facecolor()))
    for coll in ax.collections:
        for fc in np.atleast_2d(coll.get_facecolor()):
            if len(fc):
                used.add(to_hex(fc))
    return used


def _free_colour(ax, settings):
    """The first colour of the set that the main layer does not use."""
    used = _used_colours(ax)
    for c in list(settings.get("colours") or []) + palette(settings):
        if to_hex(c) not in used:
            return c
    return "black"


def _main_handle(ax, kind):
    """One artist that stands for the main layer in the legend."""
    pools = {"lines": ax.lines, "bars": ax.patches, "dots": ax.collections}
    for pool in (pools.get(kind), ax.lines, ax.patches, ax.collections):
        if pool:
            return pool[0]
    return None


def _draw_second_y(ctx, y, drawn_as, label, colour):
    last = ctx.last
    if last is None or last.get("x") is None:
        raise PlotsimpleError(
            ".second_y() adds a second y-axis to a chart, so it needs a chart to sit on.\n"
            'Draw one first, for example:\n  chart.lines(x="month", y="temperature")\n'
            '  chart.second_y("rain", drawn_as="bars", label="Rain (mm)")'
        )
    if last.get("sideways"):
        raise PlotsimpleError(
            ".second_y() cannot be used with sideways bars, because they have no y numbers to share.\n"
            "Remove sideways=True, or draw the second column as its own chart."
        )
    ax = ctx.ax
    table = last["table"].copy()
    x = last["x"]
    if is_listlike(y):
        values = list(y)
        if len(values) != len(table):
            raise PlotsimpleError(
                f".second_y() got {len(values)} values, but the chart under it has {len(table)}.\n"
                "Give one value for each row."
            )
        table["__second_y"] = values
        ycol, yname = "__second_y", label or "second y"
    else:
        if y not in table.columns:
            raise column_error(y, table.columns)
        ycol, yname = y, y

    colour = colour or _free_colour(ax, ctx.settings)
    xs = table[x]
    main_is_bars = last.get("kind") == "bars"
    categorical = main_is_bars or not (
        pd.api.types.is_numeric_dtype(xs) or pd.api.types.is_datetime64_any_dtype(xs))
    if categorical:
        # bars sit at 0, 1, 2 ... with names under them: put each value above its own name
        names = [t.get_text() for t in ax.get_xticklabels()]
        ticks = list(ax.get_xticks())
        grouped = table.groupby(xs.astype(str), sort=False)[ycol].mean()
        if any(names):
            missing = [g for g in grouped.index if g not in names]
            if missing:
                raise PlotsimpleError(
                    f'.second_y() could not find "{missing[0]}" on the x-axis of the chart under it.')
            pos = np.array([ticks[names.index(g)] for g in grouped.index], dtype=float)
        else:
            pos = np.arange(len(grouped), dtype=float)
        order = np.argsort(pos)
        px, py = pos[order], grouped.to_numpy()[order]
        width = 0.8
    elif drawn_as == "dots":
        px, py = xs.to_numpy(), table[ycol].to_numpy()
        width = None
    else:
        grouped = table.groupby(x, sort=True)[ycol].mean()  # one value per x, like .lines()
        px, py = grouped.index.to_numpy(), grouped.to_numpy()
        gaps = np.diff(np.unique(px)) if len(px) > 1 else np.array([])
        width = 0.8 * (gaps.min() if len(gaps) else 1)

    twin = ax.twinx()
    ctx.store["second_y"] = twin
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    if drawn_as == "lines":
        handle = twin.plot(px, py, color=colour, marker="o" if categorical else None, label=yname)[0]
    elif drawn_as == "dots":
        handle = twin.scatter(px, py, color=colour, label=yname)
    else:
        if main_is_bars:
            # make room: the main bars keep the left half of their place, the new bars take the right half
            for patch in ax.patches:
                patch.set_width(patch.get_width() / 2)
            for line in ax.lines:  # error bars move with their bars
                line.set_xdata(np.asarray(line.get_xdata(), dtype=float) - width / 4)
            handle = twin.bar(px, py, width=width / 2, align="edge", color=colour, label=yname)
        else:
            handle = twin.bar(px, py, width=width, color=colour, alpha=0.6, label=yname)
        # keep the main layer in front of the new bars
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
    twin.set_ylabel(label or yname, color=colour)
    twin.tick_params(axis="y", colors=colour)
    twin.spines["right"].set_color(colour)

    # one legend for both axes, on whichever axes is drawn on top
    old = ax.get_legend()
    if old is not None:
        handles = list(old.legend_handles)
        labels = [t.get_text() for t in old.texts]
        title = old.get_title().get_text() or None
        old.remove()
    else:
        main = _main_handle(ax, last.get("kind"))
        handles = [main] if main is not None else []
        labels = [str(last.get("y") or "")] if main is not None else []
        title = None
    handles.append(handle)
    labels.append(label or yname)
    top = ax if ax.get_zorder() > twin.get_zorder() else twin
    top.legend(handles, labels, title=title, loc=_emptiest_corner([ax, twin]))


def _points_in_axes(ax):
    """Points (in 0-1 axes units) covered by what is drawn on ax: line points, dots, bar areas."""
    ax.get_xlim(), ax.get_ylim()  # make matplotlib finish its automatic limits first
    pts = []
    for line in ax.lines:
        xy = np.column_stack([np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)])
        xy = xy[~np.isnan(xy).any(axis=1)]
        for a, b in zip(xy[:-1], xy[1:]):  # fill in between the points, too
            pts.extend(a + (b - a) * t for t in np.linspace(0, 1, 12))
        pts.extend(xy)
    for coll in ax.collections:
        offsets = np.asarray(coll.get_offsets(), float)
        if offsets.ndim == 2 and len(offsets) and coll.get_offset_transform() == ax.transData:
            pts.extend(offsets)
    for patch in ax.patches:
        x0, y0, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
        for fx in np.linspace(0, 1, 5):
            for fy in np.linspace(0, 1, 8):
                pts.append((x0 + fx * w, y0 + fy * h))
    if not pts:
        return np.empty((0, 2))
    data = np.asarray(pts, float)
    return ax.transAxes.inverted().transform(ax.transData.transform(data))


def _emptiest_corner(axes):
    """matplotlib's loc="best" only sees one axes; this looks at both sides of a second_y chart."""
    pts = np.concatenate([_points_in_axes(a) for a in axes])
    boxes = {
        "upper left": (0, 0.7, 0.4, 1), "upper right": (0.6, 0.7, 1, 1), "upper center": (0.3, 0.7, 0.7, 1),
        "lower left": (0, 0, 0.4, 0.3), "lower right": (0.6, 0, 1, 0.3), "lower center": (0.3, 0, 0.7, 0.3),
        "center left": (0, 0.35, 0.4, 0.65), "center right": (0.6, 0.35, 1, 0.65),
    }

    def crowd(box):
        x0, y0, x1, y1 = box
        return int(np.sum((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)))

    return min(boxes, key=lambda k: crowd(boxes[k]))  # ties: the first, "upper left"


class AxesWords:
    # -------------------------------------------------------------- ranges
    def _range_word(self, axis, low, high):
        step = f"range_{axis}"
        example = f"chart.{step}(0, 100), or chart.{step}(0, None) to fix only the low end"
        if low is None and high is None:
            raise PlotsimpleError(f".{step}() needs at least one end.\nFor example: {example}")
        _check_value(step, "low", low, example)
        _check_value(step, "high", high, example)
        if low is not None and high is not None:
            try:
                wrong = not (low < high)
            except TypeError:
                raise PlotsimpleError(
                    f".{step}() needs both ends of the same kind (two numbers or two dates).") from None
            if wrong:
                flip = f"flip_{axis}"
                raise PlotsimpleError(
                    f".{step}({low!r}, {high!r}): the low end must be smaller than the high end.\n"
                    f"Write chart.{step}({high!r}, {low!r}). To turn the axis around, use chart.{flip}()."
                )
        if (isinstance(low, numbers.Real) and low <= 0
                and any(s.name == f"log_scale_{axis}" for s in self._steps)):
            raise PlotsimpleError(
                f".{step}() starts at {low}, but this chart has .log_scale_{axis}(), "
                "which cannot show 0 or negative numbers.\n"
                f"Start above 0, for example chart.{step}(1, {high!r}), or use None for the low end."
            )
        return self._replace(step, "axes", _range, axis=axis, low=low, high=high)

    @friendly
    def range_x(self, low=None, high=None):
        """From where to where the x-axis goes. chart.range_x(0, 10)

        Use None to fix only one end: chart.range_x(0, None) starts at 0 and lets the top choose itself.
        """
        return self._range_word("x", low, high)

    @friendly
    def range_y(self, low=None, high=None):
        """From where to where the y-axis goes. chart.range_y(0, 100)

        Use None to fix only one end: chart.range_y(None, 100) stops at 100.
        """
        return self._range_word("y", low, high)

    # -------------------------------------------------------------- log scales
    def _log_word(self, axis):
        for s in self._steps:
            low = s.kwargs.get("low")
            if s.name == f"range_{axis}" and isinstance(low, numbers.Real) and low <= 0:
                raise PlotsimpleError(
                    f".log_scale_{axis}() cannot show 0 or negative numbers, but .range_{axis}() starts at "
                    f"{low}.\nChange the range to start above 0, for example "
                    f"chart.range_{axis}(1, {s.kwargs['high']!r})."
                )
        # The scale is set before drawing (so seaborn works in log space and later ticks and
        # ranges are not reset by it); the data are checked after drawing.
        self._replace(f"log_scale_{axis}", "setup", _set_log, axis=axis)
        return self._replace(f"log_scale_{axis}_check", "axes", _check_log, axis=axis)

    @friendly
    def log_scale_x(self):
        """A log scale on the x-axis: 1, 10, 100, 1000 are evenly spaced. chart.log_scale_x()"""
        return self._log_word("x")

    @friendly
    def log_scale_y(self):
        """A log scale on the y-axis: 1, 10, 100, 1000 are evenly spaced. chart.log_scale_y()"""
        return self._log_word("y")

    # -------------------------------------------------------------- ticks
    def _ticks_word(self, axis, values, labels):
        step = f"ticks_{axis}"
        if len(values) == 1 and is_listlike(values[0]):
            values = tuple(values[0])
        if not values:
            raise PlotsimpleError(
                f".{step}() needs the places where ticks go.\nFor example: chart.{step}(0, 10, 20, 30)")
        for v in values:
            if not (_is_value(v) or isinstance(v, str)):
                raise PlotsimpleError(
                    f".{step}() needs numbers (or names on the axis), not {v!r}.\n"
                    f"For example: chart.{step}(0, 10, 20, 30)")
        if labels is not None:
            if isinstance(labels, str) or not is_listlike(labels):
                raise PlotsimpleError(
                    f'labels= should be a list, for example chart.{step}(0, 50, 100, labels=["low", "mid", "high"]).')
            labels = [str(t) for t in labels]
            if len(labels) != len(values):
                raise PlotsimpleError(
                    f".{step}() got {len(values)} ticks but {len(labels)} labels. "
                    "Give one label for each tick.\n"
                    f'For example: chart.{step}(0, 50, 100, labels=["low", "mid", "high"])')
        return self._replace(step, "axes", _ticks, axis=axis, values=list(values), labels=labels)

    @friendly
    def ticks_x(self, *values, labels=None):
        """Where the marks on the x-axis go. chart.ticks_x(1, 4, 7, 10)

        labels=["Jan", "Apr", "Jul", "Oct"] writes your own words at those places.
        """
        return self._ticks_word("x", values, labels)

    @friendly
    def ticks_y(self, *values, labels=None):
        """Where the marks on the y-axis go. chart.ticks_y(0, 50, 100)

        labels=["low", "mid", "high"] writes your own words at those places.
        """
        return self._ticks_word("y", values, labels)

    # -------------------------------------------------------------- rotate
    def _rotate_word(self, axis, angle):
        if isinstance(angle, bool) or not isinstance(angle, numbers.Real):
            raise PlotsimpleError(
                f".rotate_labels_{axis}() needs an angle in degrees, for example chart.rotate_labels_{axis}(45).")
        if not -360 <= angle <= 360:
            raise PlotsimpleError(f"The angle should be between -360 and 360 degrees, not {angle}. Try 45 or 90.")
        # "text" phase: runs after ticks_x/ticks_y, so the new tick labels are turned as well
        return self._replace(f"rotate_labels_{axis}", "text", _rotate, axis=axis, angle=float(angle))

    @friendly
    def rotate_labels_x(self, angle=45):
        """Turn the names under the x-axis so long ones do not bump into each other.

        chart.rotate_labels_x() turns them 45 degrees; chart.rotate_labels_x(90) stands them up.
        """
        return self._rotate_word("x", angle)

    @friendly
    def rotate_labels_y(self, angle=45):
        """Turn the names beside the y-axis. chart.rotate_labels_y(30)"""
        return self._rotate_word("y", angle)

    # -------------------------------------------------------------- flip
    @friendly
    def flip_x(self):
        """Turn the x-axis around, so numbers grow from right to left. chart.flip_x()"""
        return self._replace("flip_x", "axes", _flip, axis="x")

    @friendly
    def flip_y(self):
        """Turn the y-axis upside down, for example for depth going down. chart.flip_y()"""
        return self._replace("flip_y", "axes", _flip, axis="y")

    # -------------------------------------------------------------- second y-axis
    @friendly(renamed={"chart": "drawn_as"})
    def second_y(self, y, drawn_as="lines", label=None, colour=None):
        """A second y-axis on the right, for a column measured in other units.

        chart.lines(x="month", y="temperature")
        chart.second_y("rain", drawn_as="bars", label="Rain (mm)")
        (with ps.example("weather"))

        drawn_as= can be "lines", "bars" or "dots". It shares the x-axis with the chart before it.
        """
        self._check_choice("second_y", "drawn_as", drawn_as, SECOND_KINDS)
        if y is None or not (isinstance(y, str) or is_listlike(y)):
            raise PlotsimpleError(
                f'.second_y() needs a column name, for example chart.second_y("rain"), not {y!r}.')
        if isinstance(y, str):
            check(self._data, y=y)
        if label is not None:
            label = str(label)
        if colour is not None and not (isinstance(colour, str) and is_color_like(colour)):
            raise PlotsimpleError(
                f'.second_y() needs one colour name for colour=, such as "red" or "green", not {colour!r}.')
        return self._replace("second_y", "extra", _draw_second_y, y=y, drawn_as=drawn_as, label=label,
                             colour=colour)
