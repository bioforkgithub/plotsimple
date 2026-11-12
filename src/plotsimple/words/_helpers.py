"""Small helpers shared by the word files."""

from __future__ import annotations

import pandas as pd
from matplotlib.colors import is_color_like

from .._data import frame_for, is_listlike
from .._errors import column_error
from .._style import COLOURS


def colour_role(data: pd.DataFrame, colour):
    """colour= can be a column (one colour per group) or a colour name ("blue").

    Returns (group, fixed): group is a column name/list to split colours by,
    fixed is a single colour. A column with the same name as a colour wins.
    """
    if colour is None:
        return None, None
    if is_listlike(colour):
        return colour, None
    if isinstance(colour, str) and colour in data.columns:
        return colour, None
    if is_color_like(colour):
        return None, colour
    raise column_error(colour, data.columns)


def palette_for(table: pd.DataFrame, group: str | None, settings: dict | None = None):
    """Colours for each group: chart.colours(...) if the user set them, else the default set."""
    if group is None:
        return None
    base = list((settings or {}).get("colours") or COLOURS)
    n = table[group].nunique(dropna=True)
    return (base * (n // len(base) + 1))[:n]


def default_colour(settings: dict | None = None, fixed=None):
    """The colour for a chart without groups: colour="red" > chart.colour("red") > first default."""
    if fixed is not None:
        return fixed
    settings = settings or {}
    if settings.get("colour"):
        return settings["colour"]
    if settings.get("colours"):
        return settings["colours"][0]
    return COLOURS[0]


def single_list(x, y):
    """chart.lines([3, 5, 2]) means: these are the y values, x is 1, 2, 3 ..."""
    if y is None and is_listlike(x):
        return list(range(1, len(x) + 1)), x
    return x, y


def check(data: pd.DataFrame, **roles):
    """Check columns/lists straight away so the error points at the user's line."""
    frame_for(data, **roles)
