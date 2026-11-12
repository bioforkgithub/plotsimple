"""Colours, legends and the overall look."""

from __future__ import annotations

from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

from .._data import is_listlike
from .._errors import PlotsimpleError, friendly, suggest
from .._style import STYLES

# Colour names we suggest from when a name is misspelt ("bleu" -> "blue").
_COLOUR_NAMES = sorted(mcolors.CSS4_COLORS)


def check_colour(value, step: str):
    """Return `value` if matplotlib can use it as a colour, else raise a friendly error."""
    if not isinstance(value, bool) and mcolors.is_color_like(value):
        return value
    lines = [f".{step}() does not know the colour {value!r}."]
    if isinstance(value, str):
        guess = suggest(value.lower().replace(" ", ""), _COLOUR_NAMES, n=1)
        if guess:
            lines.append(f'Did you mean "{guess[0]}"?')
    lines.append('Use a colour name such as "blue", "orange" or "darkgreen", or a code such as "#0072B2".')
    raise PlotsimpleError("\n".join(lines))


# ---------------------------------------------------------------- legends
def _twins(ctx) -> list:
    """Extra y-axes (chart.second_y) that may carry legend entries."""
    found = []
    stored = ctx.store.get("second_y")
    if isinstance(stored, dict):
        stored = list(stored.values())
    for item in stored if isinstance(stored, (list, tuple)) else [stored]:
        if hasattr(item, "get_legend_handles_labels"):
            found.append(item)
    for ax in ctx.axes:  # twins of every panel, not just the last one drawn
        siblings = getattr(ax, "_twinned_axes", None)
        if siblings is not None:
            found.extend(siblings.get_siblings(ax))
    unique = []
    for ax in found:
        if ax not in unique and ax not in ctx.axes:
            unique.append(ax)
    return unique


def _blank():
    return Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)


def _collect_entries(ctx):
    """Take every legend entry from all panels (and twin axes) and remove their legends."""
    handles, labels, group_title = [], [], ""
    for ax in list(ctx.axes) + _twins(ctx):
        found_h, found_l = [], []
        leg = ax.get_legend()
        if leg is not None:
            found_h = list(leg.legend_handles)
            found_l = [t.get_text() for t in leg.get_texts()]
            group_title = group_title or leg.get_title().get_text()
            leg.remove()
        more_h, more_l = ax.get_legend_handles_labels()
        for h, label in zip(found_h + more_h, found_l + more_l):
            if not label or label.startswith("_") or label in labels:
                continue
            handles.append(h if h is not None else _blank())
            labels.append(label)
    return handles, labels, group_title


def _take_text(artist):
    """Hide a figure title/label so the legend can show it instead. Returns (text, font)."""
    font = dict(size=artist.get_fontsize(), weight=artist.get_fontweight())
    artist.set_visible(False)
    artist.set_in_layout(False)
    return artist.get_text(), font


def _visible_fig_text(fig, attr):
    artist = getattr(fig, attr, None)
    if artist is not None and artist.get_visible() and artist.get_text():
        return artist
    return None


_LEGEND_PLACES = {
    "top": "outside upper center",
    "bottom": "outside lower center",
    "left": "outside center left",
    "right": "outside center right",
}


def _legend(ctx, where):
    handles, labels, group_title = _collect_entries(ctx)
    if where == "hide" or not handles:
        return
    fig = ctx.fig

    # Matplotlib puts a figure legend and a figure title/label in the same margin, on top
    # of each other. So the text that would share the legend's side becomes its heading.
    heading = None
    if where == "top":
        sup = _visible_fig_text(fig, "_suptitle")
        if sup is not None:
            heading = _take_text(sup)
        elif len(ctx.axes) == 1 and ctx.ax.get_title():  # keep the title above the legend
            title = ctx.ax.title
            heading = (title.get_text(), dict(size=title.get_fontsize(), weight=title.get_fontweight()))
            ctx.ax.set_title("")
    elif where == "bottom":
        sup = _visible_fig_text(fig, "_supxlabel")
        if sup is not None:
            heading = _take_text(sup)
    elif where == "left":
        sup = _visible_fig_text(fig, "_supylabel")
        if sup is not None:  # give the y name back to the left-hand panels instead
            text, _ = _take_text(sup)
            for ax in ctx.axes:
                spec = ax.get_subplotspec()
                if spec is None or spec.is_first_col():
                    ax.set_ylabel(text)

    kw = dict(loc=_LEGEND_PLACES[where])
    if heading is not None:
        kw["title"], kw["title_fontproperties"] = heading
        if group_title:  # the group name moves to the front of the row
            handles = [_blank()] + handles
            labels = [f"{group_title}:"] + labels
    elif group_title:
        kw["title"] = group_title
    if where in ("top", "bottom"):
        kw["ncols"] = min(len(handles), 6)  # one row; wraps only when there are many entries
    fig.legend(handles, labels, **kw)


def _colour_bar_label(ctx, text):
    bar = ctx.store.get("colour_bar")
    if bar is None:
        raise PlotsimpleError(
            ".colour_bar_label() needs a chart with a colour bar,\n"
            'for example chart.heatmap() or chart.contour(x="lon", y="lat", z="depth").'
        )
    for one in bar if isinstance(bar, (list, tuple)) else [bar]:
        one.set_label(text)


class LookWords:
    @friendly
    def style(self, name):
        """The overall look: "paper" (default), "school", "presentation" or "dark"."""
        self._check_choice("style", "name", name, list(STYLES))
        return self._set(style=name)

    @friendly
    def size(self, width=8, height=6):
        """The size in inches. chart.size(width=8, height=6)"""
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise PlotsimpleError(f"{name}= should be a number bigger than 0, not {value!r}.")
        # size_set tells one_per() and pages not to choose a size themselves
        return self._set(width=float(width), height=float(height), size_set=True)

    # ------------------------------------------------------------ colours
    @friendly
    def colour(self, name):
        """One colour for the whole chart. chart.colour("blue") or chart.colour("#0072B2")"""
        return self._set(colour=check_colour(name, "colour"))

    color = colour  # American spelling, same word

    @friendly
    def colours(self, *names):
        """Your own colours for the groups, in order. chart.colours("blue", "orange", "green")"""
        if len(names) == 1 and is_listlike(names[0]):
            names = tuple(names[0])
        if not names:
            raise PlotsimpleError('.colours() needs at least one colour, for example chart.colours("blue", "orange").')
        return self._set(colours=[check_colour(n, "colours") for n in names])

    colors = colours  # American spelling, same word

    # ------------------------------------------------------------ grid
    @friendly
    def show_grid(self):
        """Grid lines across the chart in both directions. chart.show_grid()"""
        return self._set(grid=True)

    @friendly
    def hide_grid(self):
        """No grid lines. chart.hide_grid()"""
        return self._set(grid=False)

    # ------------------------------------------------------------ legend
    def _legend_step(self, where):
        return self._replace("legend", "finish", _legend, scope="figure", where=where)

    @friendly
    def legend_top(self):
        """The legend above the chart, in one row. chart.legend_top()"""
        return self._legend_step("top")

    @friendly
    def legend_bottom(self):
        """The legend below the chart, in one row under the x-axis name. chart.legend_bottom()"""
        return self._legend_step("bottom")

    @friendly
    def legend_left(self):
        """The legend to the left of the chart. chart.legend_left()"""
        return self._legend_step("left")

    @friendly
    def legend_right(self):
        """The legend to the right of the chart, outside the plot. chart.legend_right()"""
        return self._legend_step("right")

    @friendly
    def hide_legend(self):
        """No legend. chart.hide_legend()"""
        return self._legend_step("hide")

    # ------------------------------------------------------------ colour bar
    @friendly
    def colour_bar_label(self, text):
        """The name beside the colour bar of a heatmap or contour. chart.colour_bar_label("Depth (m)")"""
        return self._replace("colour_bar_label", "text", _colour_bar_label, text=str(text))

    color_bar_label = colour_bar_label  # American spelling, same word
