"""Words and labels on a chart."""

from __future__ import annotations

import numbers

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.text import Annotation

from .._errors import PlotsimpleError, friendly
from .axes import tick_position

ARROW_OFFSET = 40  # points between a note and the place its arrow points at


def _title_top(ctx, text):
    if len(ctx.axes) == 1:
        ctx.ax.set_title(text)
    else:
        ctx.fig.suptitle(text, fontweight="bold")


def _label_x(ctx, text):
    if len(ctx.axes) == 1:
        ctx.ax.set_xlabel(text)
    else:
        for ax in ctx.axes:
            ax.set_xlabel("")
        ctx.fig.supxlabel(text, fontweight="bold")


def _label_y(ctx, text):
    if len(ctx.axes) == 1:
        ctx.ax.set_ylabel(text)
    else:
        for ax in ctx.axes:
            ax.set_ylabel("")
        ctx.fig.supylabel(text, fontweight="bold")


# ---------------------------------------------------------------- caption
def _caption_bottom(ctx, text):
    style = dict(fontsize=0.8 * mpl.rcParams["font.size"], fontweight="normal", alpha=0.8)
    if len(ctx.axes) == 1:
        # Hang the caption under the x-axis label. It belongs to the axes, so constrained
        # layout makes room for it, and it moves down when labels are turned or get bigger.
        ctx.ax.annotate(text, xy=(0.5, 0), xycoords=ctx.ax.xaxis.label, xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top",
                        annotation_clip=False, **style)
        return
    # Several panels: the figure's bottom line (supxlabel) is the only text that constrained
    # layout keeps below every panel.
    shared = ctx.fig.get_supxlabel()
    if not shared:
        ctx.fig.supxlabel(text, **style)
        return
    # Both a shared x label and a caption: the label gets an empty second line and the
    # caption sits in it, so layout has made room for both.
    label = ctx.fig.supxlabel(shared + "\n", fontweight="bold")
    ctx.fig.add_artist(Annotation(text, xy=(0.5, 0), xycoords=label, ha="center", va="bottom",
                                  annotation_clip=False, **style))


# ---------------------------------------------------------------- note
def _note(ctx, text, x, y, arrow):
    ax = ctx.ax
    # a soft background in the chart's own colour keeps the words readable over lines and bars
    back = dict(boxstyle="round,pad=0.2", fc=mpl.rcParams["axes.facecolor"], ec="none", alpha=0.85)
    px, py = tick_position(ax, "x", x), tick_position(ax, "y", y)
    if not arrow:
        ax.annotate(text, xy=(px, py), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", bbox=back)
        return
    # Put the text on the side that points into the chart, so it stays inside.
    data = (ax.xaxis.convert_units(px), ax.yaxis.convert_units(py))
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform(data))
    dx = -ARROW_OFFSET if fx > 0.6 else ARROW_OFFSET
    dy = -ARROW_OFFSET if fy > 0.6 else ARROW_OFFSET
    ax.annotate(text, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom", bbox=back,
                arrowprops=dict(arrowstyle="->", color=mpl.rcParams["text.color"], lw=1.2,
                                shrinkA=2, shrinkB=3))


# ---------------------------------------------------------------- numbers on bars
def _text_colour_on(face):
    r, g, b = to_rgb(face)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def _numbers_on_bars(ctx, decimals):
    ax = ctx.ax
    bars = [c for c in ax.containers if isinstance(c, mpl.container.BarContainer)]
    if not bars:
        raise PlotsimpleError(
            ".numbers_on_bars() writes the value on each bar, but this chart has no bars.\n"
            'Add some first, for example chart.bars(x="name", y="marks").'
        )
    sideways = getattr(bars[0], "orientation", "vertical") == "horizontal"

    def where(patch):  # middle of a bar along the category axis
        if sideways:
            return round(patch.get_y() + patch.get_height() / 2, 6)
        return round(patch.get_x() + patch.get_width() / 2, 6)

    # Stacked bars share their places; bars side by side (colour=) do not.
    places = [{where(p) for p in c.patches} for c in bars]
    stacked = len(bars) > 1 and all(p == places[0] for p in places[1:])

    def text(v):
        if v is None or np.isnan(v) or (stacked and v == 0):
            return ""
        return f"{v:,.{decimals}f}"

    if stacked:
        for c in bars:
            labels = [text(v) for v in np.ravel(c.datavalues)]
            colour = _text_colour_on(c.patches[0].get_facecolor())
            for t in ax.bar_label(c, labels=labels, label_type="center"):
                t.set_color(colour)
        return

    # Error bars: write the number just past the end of the error bar, not across it.
    # (seaborn draws each error bar, caps included, as one line that stays within its bar)
    spans = []
    for line in ax.lines:
        xd, yd = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        along, value = (yd, xd) if sideways else (xd, yd)
        if len(along) and not np.all(np.isnan(along)):
            spans.append((np.nanmin(along), np.nanmax(along), value))

    def error_ends(patch):
        start = patch.get_y() if sideways else patch.get_x()
        size = patch.get_height() if sideways else patch.get_width()
        lo, hi = sorted((start, start + size))
        found = [val for a, b, val in spans if a >= lo - 1e-9 and b <= hi + 1e-9]
        return np.concatenate(found) if found else None

    for c in bars:
        plain = []
        for patch, v in zip(c.patches, np.ravel(c.datavalues)):
            lab = text(v)
            span = error_ends(patch)
            if span is None or not lab:
                plain.append(lab)
                continue
            plain.append("")
            up = v >= 0
            end = np.nanmax(span) if up else np.nanmin(span)
            if sideways:
                ax.annotate(lab, xy=(end, where(patch)), xytext=(3 if up else -3, 0),
                            textcoords="offset points", ha="left" if up else "right", va="center")
            else:
                ax.annotate(lab, xy=(where(patch), end), xytext=(0, 3 if up else -3),
                            textcoords="offset points", ha="center", va="bottom" if up else "top")
        ax.bar_label(c, labels=plain, padding=3)
    # a little more room at the end of the bars, so the numbers stay inside the chart
    ax.margins(**({"x": 0.12} if sideways else {"y": 0.12}))
    ax.autoscale_view()


# ---------------------------------------------------------------- the words
def _check_text(step, text, example):
    if text is None or (isinstance(text, str) and not text.strip()):
        raise PlotsimpleError(f".{step}() needs some words, for example {example}")
    return str(text)


class TextWords:
    @friendly
    def title_top(self, text):
        """A title above the chart. chart.title_top("My class marks")"""
        return self._replace("title_top", "text", _title_top, scope="figure", text=str(text))

    @friendly
    def label_x(self, text):
        """The name under the x-axis. chart.label_x("Student")"""
        return self._replace("label_x", "text", _label_x, scope="figure", text=str(text))

    @friendly
    def label_y(self, text):
        """The name beside the y-axis. chart.label_y("Marks")"""
        return self._replace("label_y", "text", _label_y, scope="figure", text=str(text))

    @friendly
    def caption_bottom(self, text):
        """A small line of text under the whole chart. chart.caption_bottom("Data from 2026")"""
        text = _check_text("caption_bottom", text, 'chart.caption_bottom("Data from 2026")')
        # "finish": runs after label_x, so it knows whether there is a shared x label to go under
        return self._replace("caption_bottom", "finish", _caption_bottom, scope="figure", text=text)

    @friendly
    def note(self, text, x=None, y=None, arrow=False):
        """A note on the chart at a place x, y. chart.note("Highest!", x=5, y=98)

        arrow=True puts the note a little away, with an arrow pointing at x, y.
        On bar charts x can be a name: chart.note("Best", x="Ben", y=90).
        """
        text = _check_text("note", text, 'chart.note("Highest!", x=5, y=98)')
        if x is None or y is None:
            raise PlotsimpleError(
                '.note() needs a place: x= and y=, for example chart.note("Highest!", x=5, y=98).')
        for name, v in (("x", x), ("y", y)):
            ok = isinstance(v, (numbers.Real, str)) or hasattr(v, "year")  # numbers, names, dates
            if isinstance(v, bool) or not ok:
                raise PlotsimpleError(
                    f'.note() needs a number for {name}=, not {v!r}. For example chart.note("Highest!", x=5, y=98).')
        if not isinstance(arrow, bool):
            raise PlotsimpleError(".note() needs arrow=True or arrow=False.")
        return self._add("note", "extra", _note, text=text, x=x, y=y, arrow=arrow)

    @friendly
    def numbers_on_bars(self, decimals=0):
        """Write the value of each bar on it. chart.numbers_on_bars() · decimals=1 shows 12.5"""
        if isinstance(decimals, bool) or not isinstance(decimals, numbers.Integral) or not 0 <= decimals <= 6:
            raise PlotsimpleError(
                ".numbers_on_bars() needs a whole number from 0 to 6 for decimals=, "
                "for example chart.numbers_on_bars(decimals=1).")
        return self._replace("numbers_on_bars", "extra", _numbers_on_bars, decimals=int(decimals))

    @friendly
    def font_size(self, size):
        """Make all the text bigger or smaller. chart.font_size(14) (normal is 12)"""
        if isinstance(size, bool) or not isinstance(size, numbers.Real) or size <= 0:
            raise PlotsimpleError(
                f".font_size() needs a number bigger than 0, not {size!r}. For example chart.font_size(14).")
        if not 4 <= size <= 60:
            raise PlotsimpleError(
                f"Font size {size} is very {'small' if size < 4 else 'big'}. "
                "Try a number between 8 and 24, for example chart.font_size(14).")
        return self._set(font_size=float(size))
