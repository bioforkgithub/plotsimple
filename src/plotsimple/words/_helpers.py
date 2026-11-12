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


def palette_for(table: pd.DataFrame, group: str | None):
    if group is None:
        return None
    n = table[group].nunique(dropna=True)
    return (COLOURS * (n // len(COLOURS) + 1))[:n]


def single_list(x, y):
    """chart.lines([3, 5, 2]) means: these are the y values, x is 1, 2, 3 ..."""
    if y is None and is_listlike(x):
        return list(range(1, len(x) + 1)), x
    return x, y


def check(data: pd.DataFrame, **roles):
    """Check columns/lists straight away so the error points at the user's line."""
    frame_for(data, **roles)
