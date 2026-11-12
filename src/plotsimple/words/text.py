"""Words and labels on a chart."""

from __future__ import annotations

from .._errors import friendly


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
