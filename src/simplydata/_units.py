"""Sizes in inches, centimetres, millimetres or pixels."""

from __future__ import annotations

from ._errors import SimplydataError, choice_error

# how many of each unit make one inch
PER_INCH = {"inches": 1.0, "cm": 2.54, "mm": 25.4}
UNITS = ("inches", "cm", "mm", "pixels")
SCREEN_RESOLUTION = 100  # dots per inch used to show a chart on screen
_ALIASES = {"inch": "inches", "in": "inches", "centimetres": "cm", "centimeters": "cm",
            "millimetres": "mm", "millimeters": "mm", "pixel": "pixels", "px": "pixels"}


def check_number(value, name: str, step: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value <= 0:
        raise SimplydataError(f".{step}() needs {name}= to be a number bigger than 0, not {value!r}.")
    return float(value)


def check_unit(unit, step: str) -> str:
    unit = _ALIASES.get(unit, unit) if isinstance(unit, str) else unit
    if unit not in UNITS:
        raise choice_error(step, "unit", unit, list(UNITS))
    return unit
