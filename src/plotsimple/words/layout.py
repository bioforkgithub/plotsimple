"""Several charts together, small multiples, zoom boxes and animation.

How the hard tricks work underneath (users never see this):
* one_per        sets settings["split"]; BaseChart._render_split draws one panel per group.
* side_by_side,  a Page gives every chart its own matplotlib SubFigure, so each chart keeps
  one_above_other, its own title, labels and legend, and figure-wide steps of one chart
  grid             (labels, legends) cannot land on another chart.
* zoom_box       draws the chart's data steps again into an inset axes.
* animate        draws the chart once per value of a column and saves the frames as a GIF.
"""

from __future__ import annotations

import math
import numbers

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from .. import _style
from .._core import PHASES, BaseChart, Context
from .._data import is_listlike
from .._errors import PlotsimpleError, column_error, friendly
from .look import LookWords

# Inset positions inside the main plot: [left, bottom, width, height] as parts of the plot.
_ZOOM_PLACES = {
    "upper right": [0.58, 0.56, 0.39, 0.4],
    "upper left": [0.03, 0.56, 0.39, 0.4],
    "lower right": [0.58, 0.05, 0.39, 0.4],
    "lower left": [0.03, 0.05, 0.39, 0.4],
}
# Steps that must not be drawn again inside a zoom box.
_NOT_IN_ZOOM = {"zoom_box", "second_y"}


def _draw_chart(chart, fig):
    """Draw every step of `chart` into `fig` (a Figure or a SubFigure). Returns the panels."""
    split = chart._settings["split"]
    if not split:
        ax = fig.add_subplot(projection=chart._settings["projection"])
        axes, ctx = [ax], chart._draw_panel(fig, ax)
    else:
        axes, ctx = chart._render_split(fig, split)
    chart._draw_figure_steps(fig, axes, ctx)
    return axes


def _check_bool(step, name, value):
    if not isinstance(value, bool):
        raise PlotsimpleError(f".{step}() needs {name}=True or {name}=False, not {value!r}.")


# ---------------------------------------------------------------- zoom box
def _zoom_box(ctx, chart, x_from, x_to, y_from, y_to, where):
    ax = ctx.ax
    if ax.name != "rectilinear":
        raise PlotsimpleError(".zoom_box() works on flat charts with an x and a y axis, not on polar or 3D charts.")
    inset = ax.inset_axes(_ZOOM_PLACES[where])
    sub = Context(fig=ctx.fig, ax=inset, data=ctx.data, settings=ctx.settings, axes=[inset])
    again = [s for s in chart._steps
             if s.scope == "axes" and s.phase in (PHASES["draw"], PHASES["extra"], PHASES["axes"])
             and s.name not in _NOT_IN_ZOOM]
    chart._run(again, sub)
    if inset.get_legend() is not None:
        inset.get_legend().remove()
    inset.set_xlim(x_from, x_to)
    inset.set_ylim(y_from, y_to)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_title("")
    inset.tick_params(labelsize="small")
    inset.set_facecolor(matplotlib.rcParams["axes.facecolor"])
    inset.set_zorder(5)  # above the main plot's lines and legend
    ax.indicate_inset_zoom(inset, edgecolor=matplotlib.rcParams["axes.edgecolor"])


# ---------------------------------------------------------------- animation helpers
def _frame_values(data, over):
    values = list(pd.unique(data[over].dropna()))
    try:
        return sorted(values)
    except TypeError:  # mixed types that cannot be sorted: keep the order in the table
        return values


def _label_frame(fig, axes, over, value):
    text = f"{over}: {value}"
    sup = getattr(fig, "_suptitle", None)
    if len(axes) == 1:
        title = axes[0].get_title()
        axes[0].set_title(f"{title}  ({text})" if title else text)
    elif sup is not None and sup.get_visible() and sup.get_text():
        sup.set_text(f"{sup.get_text()}  ({text})")
    else:
        fig.suptitle(text, fontweight="bold")


class LayoutWords:
    # ------------------------------------------------------------ small multiples
    @friendly
    def one_per(self, by=None, row=None, column=None, columns=None, same_x=True, same_y=True):
        """One small chart for each group. chart.one_per("class", columns=3) (with ps.example("marks"))

        row="class", column="subject" makes a table of charts: one row per class, one column
        per subject.
        same_x=False / same_y=False lets each small chart choose its own axis range.
        """
        if by is None and row is None and column is None:
            raise PlotsimpleError(
                '.one_per() needs a column to split by, for example chart.one_per("class")\n'
                'or chart.one_per(row="class", column="subject").'
            )
        if by is not None and (row is not None or column is not None):
            raise PlotsimpleError(
                '.one_per() takes either one column (chart.one_per("class"))\n'
                'or row= and column= (chart.one_per(row="class", column="subject")), not both.'
            )
        if columns is not None:
            if by is None:
                raise PlotsimpleError('columns= only works with one column, for example chart.one_per("class", columns=3).')
            if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
                raise PlotsimpleError(f"columns= should be a whole number such as 2 or 3, not {columns!r}.")
        _check_bool("one_per", "same_x", same_x)
        _check_bool("one_per", "same_y", same_y)
        data = self._data
        if data is None or len(data.columns) == 0:
            raise PlotsimpleError('.one_per() needs a table, for example ps.chart(ps.example("marks")).one_per("class").')
        for name in (by, row, column):
            if name is not None and name not in data.columns:
                raise column_error(name, data.columns)

        if by is not None:
            n = data[by].nunique(dropna=True)
            ncols = min(columns or min(n, 3), max(n, 1))
            nrows = math.ceil(n / ncols) if n else 1
        else:
            nrows = data[row].nunique(dropna=True) if row else 1
            ncols = data[column].nunique(dropna=True) if column else 1
        self._set(split=dict(by=by, row=row, column=column, columns=columns, same_x=same_x, same_y=same_y))
        if not self._settings.get("size_set"):  # about 4 x 3.3 inches for each small chart
            self._set(width=max(6.0, 4.0 * ncols + 1.0), height=max(4.5, 3.3 * nrows + 1.2))
        return self

    # ------------------------------------------------------------ zoom box
    @friendly
    def zoom_box(self, x_from, x_to, y_from, y_to, where="upper right"):
        """A small box showing part of the chart close up. chart.zoom_box(x_from=2, x_to=4, y_from=40, y_to=60)

        where="upper right" (default), "upper left", "lower right" or "lower left".
        """
        self._check_choice("zoom_box", "where", where, list(_ZOOM_PLACES))
        for name, value in (("x_from", x_from), ("x_to", x_to), ("y_from", y_from), ("y_to", y_to)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise PlotsimpleError(f".zoom_box() needs a number for {name}=, not {value!r}.")
        for a, b, axis in ((x_from, x_to, "x"), (y_from, y_to, "y")):
            if a >= b:
                raise PlotsimpleError(
                    f"{axis}_from= should be smaller than {axis}_to=, for example "
                    f"{axis}_from={min(a, b)}, {axis}_to={max(a, b)}."
                )
        return self._replace("zoom_box", "finish", _zoom_box, chart=self, x_from=x_from, x_to=x_to,
                             y_from=y_from, y_to=y_to, where=where)

    # ------------------------------------------------------------ animation
    @friendly
    def animate(self, over, seconds_per_frame=0.8, keep=False):
        """A moving chart with one picture for each value of a column.

        chart.animate(over="year") (with ps.example("growth")), then chart.save("growth.gif")
        keep=True builds the chart up: each picture also keeps the values before it.
        """
        data = self._data
        if data is None or len(data.columns) == 0:
            raise PlotsimpleError('.animate() needs a table, for example ps.chart(ps.example("growth")).animate(over="year").')
        if is_listlike(over) or over not in data.columns:
            raise column_error(over, data.columns)
        if isinstance(seconds_per_frame, bool) or not isinstance(seconds_per_frame, numbers.Real) \
                or seconds_per_frame <= 0:
            raise PlotsimpleError(f"seconds_per_frame= should be a number bigger than 0, such as 0.8, not {seconds_per_frame!r}.")
        _check_bool("animate", "keep", keep)
        if data[over].nunique(dropna=True) < 2:
            raise PlotsimpleError(f'Column "{over}" has fewer than 2 different values, so there is nothing to animate.')
        return self._set(animate=dict(over=over, seconds=float(seconds_per_frame), keep=keep))

    def _frames(self):
        anim = self._settings["animate"]
        data, over = self._data, anim["over"]
        values = _frame_values(data, over)
        for i, value in enumerate(values):
            shown = values[: i + 1] if anim["keep"] else [value]
            yield value, data[data[over].isin(shown)]

    def _fixed_limits(self):
        """Axis ranges of the chart with all the data, so frames do not jump about."""
        fig = Figure(figsize=(self._settings["width"], self._settings["height"]), layout="constrained")
        return [(ax.get_xlim(), ax.get_ylim()) for ax in _draw_chart(self, fig)]

    def _draw_frame(self, fig, value, table, limits):
        fig.clear()
        full = self._data
        self._data = table
        try:
            axes = _draw_chart(self, fig)
        finally:
            self._data = full
        for ax, (xlim, ylim) in zip(axes, limits):
            if ax.name == "rectilinear":
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        _label_frame(fig, axes, self._settings["animate"]["over"], value)

    def _render(self):
        """An animated chart shown as a still picture (in Jupyter or as .png) shows its last frame."""
        if not self._settings.get("animate"):
            return super()._render()
        if self._cache is not None:
            return self._cache
        s = self._settings
        with _style.context(s):
            limits = self._fixed_limits()
            fig = Figure(figsize=(s["width"], s["height"]), layout="constrained")
            *_, (value, table) = self._frames()
            self._draw_frame(fig, value, table, limits)
        self._cache = fig
        return fig

    def _save_animation(self, path):
        if not self._settings.get("animate"):
            return super()._save_animation(path)
        from matplotlib.animation import PillowWriter

        s = self._settings
        seconds = s["animate"]["seconds"]
        with _style.context(s), matplotlib.rc_context({"savefig.bbox": None}):  # every frame the same size
            limits = self._fixed_limits()
            fig = Figure(figsize=(s["width"], s["height"]), layout="constrained")
            writer = PillowWriter(fps=1.0 / seconds)
            with writer.saving(fig, str(path), dpi=100):
                for value, table in self._frames():
                    self._draw_frame(fig, value, table, limits)
                    writer.grab_frame()
        return self


# ---------------------------------------------------------------- pages of several charts
class Page(BaseChart):
    """Several charts on one page. Make one with ps.side_by_side, ps.one_above_other or ps.grid."""

    def __init__(self, charts, rows, columns, same_x=False, same_y=False, how="grid"):
        self._charts = list(charts)
        self._rows, self._columns, self._how = rows, columns, how
        self._same_x, self._same_y = same_x, same_y
        self._data = pd.DataFrame()
        self._steps = []
        styles = {c._settings["style"] for c in self._charts}
        self._settings = {
            "style": styles.pop() if len(styles) == 1 else "paper",
            "style_set": False,  # True after page.style(...): then every chart uses it
            "font_size": None,
            "grid": None,
            "title": None,
            "size_set": False,
            "width": None,
            "height": None,
        }
        self._cache = None

    size = LookWords.size

    @friendly
    def style(self, name):
        """One look for every chart on the page: "paper", "school", "presentation" or "dark"."""
        LookWords.style(self, name)
        return self._set(style_set=True)

    @friendly
    def title_top(self, text):
        """A title above the whole page. page.title_top("Marks in both classes")"""
        return self._set(title=str(text))

    def _page_size(self):
        s = self._settings
        if s["size_set"]:
            return s["width"], s["height"]
        # each cell gets about 3/4 of its chart's own size (8 x 6 -> 6 x 4.5)
        w = max(c._settings["width"] for c in self._charts) * 0.75
        h = max(c._settings["height"] for c in self._charts) * 0.75
        return min(w * self._columns, 20.0), min(h * self._rows + (0.5 if s["title"] else 0), 20.0)

    def _render(self):
        s = self._settings
        with _style.context(s):
            fig = Figure(figsize=self._page_size(), layout="constrained")
            cells = fig.subfigures(self._rows, self._columns, squeeze=False)
            panels = []
            for chart, cell in zip(self._charts, cells.flat):
                look = dict(chart._settings)
                if s["style_set"]:
                    look["style"] = s["style"]
                with _style.context(look):
                    cell.set_facecolor(matplotlib.rcParams["figure.facecolor"])
                    panels.extend(ax for ax in _draw_chart(chart, cell) if ax.name == "rectilinear")
            if self._same_x:
                _share(panels, "x")
            if self._same_y:
                _share(panels, "y")
            if s["title"]:
                fig.suptitle(s["title"], fontweight="bold", fontsize=matplotlib.rcParams["axes.titlesize"] * 1.15)
        return fig  # not cached: the charts on the page may still change

    def _save_animation(self, path):
        raise PlotsimpleError("A page of charts cannot be saved as .gif. Save it as .png or .pdf instead.")

    def __repr__(self):
        return f"<plotsimple page: {len(self._charts)} charts, {self._how}>"


def _share(axes, which):
    """Give all panels the same x or y range, wide enough for every panel's data."""
    if len(axes) < 2:
        return
    lims = [a.get_xlim() if which == "x" else a.get_ylim() for a in axes]
    low, high = min(min(lim) for lim in lims), max(max(lim) for lim in lims)
    for ax, lim in zip(axes, lims):  # a flipped axis stays flipped
        (ax.set_xlim if which == "x" else ax.set_ylim)(*((low, high) if lim[0] <= lim[1] else (high, low)))


def _page_charts(step, charts):
    if len(charts) == 1 and isinstance(charts[0], (list, tuple)):
        charts = tuple(charts[0])
    if not charts:
        raise PlotsimpleError(f"ps.{step}() needs some charts, for example ps.{step}(chart_a, chart_b).")
    for i, c in enumerate(charts, start=1):
        if isinstance(c, Page):
            raise PlotsimpleError(
                f"Chart {i} given to ps.{step}() is already a page of charts.\n"
                "Give the single charts instead, for example ps.grid(a, b, c, d, columns=2)."
            )
        if not isinstance(c, BaseChart):
            raise PlotsimpleError(
                f"ps.{step}() needs charts made with ps.chart(...), but number {i} is {type(c).__name__}."
            )
    return charts


def _check_page_options(step, same_x, same_y):
    _check_bool(step, "same_x", same_x)
    _check_bool(step, "same_y", same_y)


def side_by_side(*charts, same_x=False, same_y=False):
    """Charts next to each other. ps.side_by_side(a, b).save("both.png")

    same_y=True gives them all the same y range, so heights can be compared.
    """
    charts = _page_charts("side_by_side", charts)
    _check_page_options("side_by_side", same_x, same_y)
    return Page(charts, 1, len(charts), same_x, same_y, how="side by side")


def one_above_other(*charts, same_x=False, same_y=False):
    """Charts on top of each other. ps.one_above_other(a, b).save("both.png")

    same_x=True gives them all the same x range, which suits time series.
    """
    charts = _page_charts("one_above_other", charts)
    _check_page_options("one_above_other", same_x, same_y)
    return Page(charts, len(charts), 1, same_x, same_y, how="one above the other")


def grid(*charts, columns=2, same_x=False, same_y=False):
    """Charts in rows and columns. ps.grid(a, b, c, d, columns=2)"""
    charts = _page_charts("grid", charts)
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise PlotsimpleError(f"columns= should be a whole number such as 2 or 3, not {columns!r}.")
    _check_page_options("grid", same_x, same_y)
    columns = min(columns, len(charts))
    return Page(charts, math.ceil(len(charts) / columns), columns, same_x, same_y, how=f"grid of {columns} columns")
