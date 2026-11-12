"""Colours and the overall look."""

from __future__ import annotations

from .._errors import PlotsimpleError, friendly
from .._style import STYLES


class LookWords:
    @friendly
    def style(self, name):
        """The overall look: "paper" (default), "school", "presentation" or "dark"."""
        self._check_choice("style", "name", name, list(STYLES))
        return self._set(style=name)

    @friendly
    def size(self, width=8, height=6):
        """The size of the chart in inches. chart.size(width=8, height=6)"""
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, (int, float)) or value <= 0:
                raise PlotsimpleError(f"{name}= should be a number bigger than 0, not {value!r}.")
        return self._set(width=float(width), height=float(height))
