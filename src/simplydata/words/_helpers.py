"""Small helpers shared by the word files."""

from __future__ import annotations

import pandas as pd
from matplotlib.colors import is_color_like

from .._data import frame_for, is_listlike
from .._errors import column_error
from .._style import palette


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
    base = list((settings or {}).get("colours") or palette(settings))
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
    return palette(settings)[0]


def single_list(x, y):
    """chart.lines([3, 5, 2]) means: these are the y values, x is 1, 2, 3 ..."""
    if y is None and is_listlike(x):
        return list(range(1, len(x) + 1)), x
    return x, y


def check(data: pd.DataFrame, **roles):
    """Check columns/lists straight away so the error points at the user's line."""
    frame_for(data, **roles)


def check_transparency(word: str, value):
    """transparency= is how see-through marks are: 0 is solid, 0.5 half see-through.

    Returns the value as a float (or None). matplotlib's alpha is 1 - transparency.
    """
    import numbers

    from .._errors import SimplydataError

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0 <= value <= 1:
        raise SimplydataError(
            f"transparency= in .{word}() should be a number from 0 (solid) to 1 (see-through), not {value!r}.\n"
            f"For example transparency=0.5 makes the marks half see-through."
        )
    if value == 1:
        raise SimplydataError(
            f"transparency=1 in .{word}() would make the marks invisible. Try 0.5 or 0.7.")
    return float(value)


def opacity(transparency, normal=None):
    """The alpha matplotlib needs: 1 - transparency, or `normal` when none was given."""
    return normal if transparency is None else 1 - transparency


def check_thickness(word: str, value):
    """thickness= is the width of lines in points (normal lines are 2, thin ones 1)."""
    import numbers

    from .._errors import SimplydataError

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not value > 0:
        raise SimplydataError(
            f"thickness= in .{word}() should be a number bigger than 0 (the width of the line in points), "
            f"not {value!r}.\nFor example thickness=3 for a thick line, thickness=0.5 for a thin one."
        )
    if value > 20:
        raise SimplydataError(
            f"thickness={value} in .{word}() is very thick. Lines are usually 0.5 to 5 points wide.")
    return float(value)


def faces_see_through(artists, transparency):
    """Make the filled part of shapes see-through, keeping their outlines solid."""
    from matplotlib.colors import to_rgba_array

    if transparency is None:
        return
    alpha = 1 - transparency
    for artist in artists:
        faces = to_rgba_array(artist.get_facecolor())
        if not len(faces):
            continue
        faces[:, 3] = faces[:, 3] * alpha
        artist.set_facecolor(faces if len(faces) > 1 else faces[0])
