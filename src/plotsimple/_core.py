"""The heart of plotsimple: a chart records steps and draws them when needed.

Why record instead of drawing straight away?
* the order of a user's lines does not matter (legend before bars still works)
* style and size can be changed at any time
* the same steps can be drawn again on many panels (one_per, side_by_side)

Every public step:
1. checks its input straight away, so mistakes are reported on the line that made them
2. records a Step (what to draw, in which phase)
3. returns the chart, so steps can also be chained on one line
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from matplotlib.figure import Figure

from . import _style
from ._data import load
from ._errors import PlotsimpleError, choice_error, missing_word_error, suggest

# Phases decide drawing order, whatever order the user wrote the steps in.
PHASES = {
    "setup": 0,   # things that must exist before drawing
    "draw": 1,    # bars, lines, dots ... (kept in the user's order, so later layers sit on top)
    "extra": 2,   # fit lines, error bars, reference lines, shading, notes
    "axes": 3,    # ranges, log scales, ticks
    "text": 4,    # titles and labels
    "finish": 5,  # legend and other last touches
}

SAVE_FORMATS = ("png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff", "gif")


@dataclass
class Step:
    name: str
    phase: int
    func: Callable
    kwargs: dict
    scope: str = "axes"  # "axes": once per panel; "figure": once per figure
    order: int = 0


@dataclass
class Context:
    """Everything a step needs while drawing."""

    fig: Figure
    ax: object
    data: pd.DataFrame
    settings: dict
    axes: list = field(default_factory=list)  # all panels (figure-scope steps use this)
    panel_title: str | None = None
    last: dict | None = None  # the most recent data layer: kind, table, x, y, colour
    store: dict = field(default_factory=dict)  # free space for steps (e.g. second y-axis)


class BaseChart:
    """A chart. Make one with ps.chart(...)."""

    def __init__(self, data=None, sheet=None, big_data: bool = False):
        self._data = load(data, sheet=sheet, big_data=big_data)
        self._steps: list[Step] = []
        self._settings = {
            "style": "paper",
            "width": 8.0,
            "height": 6.0,
            "font_size": None,
            "grid": None,
            "split": None,  # set by one_per(): dict(by=, row=, column=, columns=)
            "projection": None,  # "polar" or "3d" for charts that need special axes
        }
        self._cache: Figure | None = None

    # ------------------------------------------------------------------ recording
    def _add(self, name: str, phase: str, func: Callable, scope: str = "axes", **kwargs):
        self._steps.append(Step(name, PHASES[phase], func, kwargs, scope, len(self._steps)))
        self._cache = None
        return self

    def _replace(self, name: str, phase: str, func: Callable, scope: str = "axes", **kwargs):
        """Like _add, but a step with this name replaces an earlier one (e.g. title_top)."""
        self._steps = [s for s in self._steps if s.name != name]
        return self._add(name, phase, func, scope, **kwargs)

    def _set(self, **settings):
        self._settings.update(settings)
        self._cache = None
        return self

    @staticmethod
    def _check_choice(step: str, option: str, value, choices):
        if value not in choices:
            raise choice_error(step, option, value, list(choices))

    # ------------------------------------------------------------------ drawing
    def _run(self, steps, ctx: Context):
        for step in sorted(steps, key=lambda s: (s.phase, s.order)):
            try:
                step.func(ctx, **step.kwargs)
            except PlotsimpleError:
                raise
            except Exception as exc:  # turn library errors into a plain message
                raise PlotsimpleError(
                    f"Something went wrong while drawing .{step.name}():\n{exc}"
                ) from exc

    def _draw_panel(self, fig, ax, data=None, panel_title=None, axes=None) -> Context:
        """Draw all panel steps on one set of axes. Used by layouts as well."""
        ctx = Context(
            fig=fig,
            ax=ax,
            data=self._data if data is None else data,
            settings=self._settings,
            axes=axes if axes is not None else [ax],
            panel_title=panel_title,
        )
        self._run([s for s in self._steps if s.scope == "axes"], ctx)
        if panel_title is not None and not ax.get_title():
            ax.set_title(str(panel_title))
        return ctx

    def _draw_figure_steps(self, fig, axes, last_ctx: Context | None = None):
        ctx = Context(
            fig=fig,
            ax=axes[0],
            data=self._data,
            settings=self._settings,
            axes=list(axes),
            last=last_ctx.last if last_ctx else None,
            store=last_ctx.store if last_ctx else {},
        )
        self._run([s for s in self._steps if s.scope == "figure"], ctx)

    def _render(self) -> Figure:
        if self._cache is not None:
            return self._cache
        s = self._settings
        with _style.context(s):
            fig = Figure(figsize=(s["width"], s["height"]), layout="constrained")
            split = s["split"]
            if not split:
                ax = fig.add_subplot(projection=s["projection"])
                ctx = self._draw_panel(fig, ax)
                self._draw_figure_steps(fig, [ax], ctx)
            else:
                axes, ctx = self._render_split(fig, split)
                self._draw_figure_steps(fig, axes, ctx)
        self._cache = fig
        return fig

    def _render_split(self, fig, split):
        data = self._data
        row, column, by = split.get("row"), split.get("column"), split.get("by")
        share = dict(sharex=split.get("same_x", True), sharey=split.get("same_y", True))
        ctx = None
        if by is not None:
            groups = list(pd.unique(data[by].dropna()))
            ncols = split.get("columns") or min(len(groups), 3)
            nrows = math.ceil(len(groups) / ncols)
            grid = fig.subplots(nrows, ncols, squeeze=False, subplot_kw={"projection": self._settings["projection"]}, **share)
            flat = list(grid.flat)
            for ax, g in zip(flat, groups):
                ctx = self._draw_panel(fig, ax, data[data[by] == g], panel_title=g, axes=flat)
            for ax in flat[len(groups):]:
                ax.set_visible(False)
            return flat[: len(groups)], ctx
        rows = list(pd.unique(data[row].dropna())) if row else [None]
        cols = list(pd.unique(data[column].dropna())) if column else [None]
        grid = fig.subplots(len(rows), len(cols), squeeze=False, subplot_kw={"projection": self._settings["projection"]}, **share)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                mask = pd.Series(True, index=data.index)
                parts = []
                if row:
                    mask &= data[row] == r
                    parts.append(f"{row} = {r}")
                if column:
                    mask &= data[column] == c
                    parts.append(f"{column} = {c}")
                ctx = self._draw_panel(fig, grid[i][j], data[mask], panel_title=", ".join(parts), axes=list(grid.flat))
        return list(grid.flat), ctx

    # ------------------------------------------------------------------ output
    def save(self, file_name, quality: str = "normal"):
        """Save the chart. The file name decides the format: "marks.png", "marks.pdf"."""
        self._check_choice("save", "quality", quality, ("normal", "high"))
        path = Path(file_name)
        fmt = path.suffix.lower().lstrip(".")
        if not fmt:
            raise PlotsimpleError(
                f'Add a file type to the name, for example "{path.name}.png" or "{path.name}.pdf".'
            )
        if fmt not in SAVE_FORMATS:
            raise choice_error("save", "file type", fmt, SAVE_FORMATS)
        if path.parent and not path.parent.exists():
            raise PlotsimpleError(f'The folder "{path.parent}" does not exist. Create it first.')
        if fmt == "gif":
            return self._save_animation(path)
        fig = self._render()
        dpi = 600 if quality == "high" else 300
        with _style.context(self._settings):
            fig.savefig(path, dpi=dpi)
        return self

    def _save_animation(self, path):
        raise PlotsimpleError('To save a .gif, first make an animation with chart.animate(over="column").')

    def show(self):
        """Show the chart in a window. (In Jupyter, VS Code and Spyder charts appear by themselves.)"""
        import matplotlib.pyplot as plt

        self._cache = None
        fig = self._render()
        holder = plt.figure()
        manager = holder.canvas.manager
        manager.canvas.figure = fig
        fig.set_canvas(manager.canvas)
        plt.show()
        return self

    def _repr_png_(self):
        """Lets Jupyter, VS Code and Spyder show the chart automatically."""
        fig = self._render()
        buf = io.BytesIO()
        with _style.context(self._settings):
            fig.savefig(buf, format="png", dpi=100)
        return buf.getvalue()

    def __repr__(self):
        kinds = [s.name for s in self._steps if s.phase == PHASES["draw"]]
        what = ", ".join(kinds) if kinds else "empty"
        return f"<plotsimple chart: {what}>"

    # ------------------------------------------------------------------ help
    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise missing_word_error(name, self._word_names())

    def __setattr__(self, name, value):
        # chart.title = "x" would silently do nothing; tell the user what to write instead.
        if not name.startswith("_"):
            words = self._word_names()
            guesses = [w for w in words if w.startswith(name)] or suggest(name, words)
            hint = f'chart.{guesses[0]}({value!r})' if guesses else "a step such as chart.title_top(\"...\")"
            raise PlotsimpleError(
                f"Steps are written with brackets, not with =.\nInstead of chart.{name} = ..., write {hint}"
            )
        object.__setattr__(self, name, value)
