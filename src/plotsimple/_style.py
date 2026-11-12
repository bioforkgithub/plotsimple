"""The looks a chart can have. "paper" is the default."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib

# Okabe-Ito colours: safe for people with colour blindness and clear in print.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

_BASE = {
    "axes.prop_cycle": matplotlib.cycler(color=COLOURS),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "image.cmap": "viridis",
}

STYLES = {
    "paper": {},
    "school": {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "axes.linewidth": 1.5,
    },
    "presentation": {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "lines.linewidth": 3.0,
    },
    "dark": {
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#1e1e1e",
        "savefig.facecolor": "#1e1e1e",
        "axes.edgecolor": "#dddddd",
        "axes.labelcolor": "#eeeeee",
        "text.color": "#eeeeee",
        "xtick.color": "#dddddd",
        "ytick.color": "#dddddd",
        "grid.color": "#555555",
        "axes.prop_cycle": matplotlib.cycler(
            color=["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7", "#D55E00", "#0072B2", "#FFFFFF"]
        ),
    },
}

_FONT_KEYS = ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")


def rc_for(settings: dict) -> dict:
    rc = dict(_BASE)
    rc.update(STYLES[settings.get("style", "paper")])
    size = settings.get("font_size")
    if size:
        scale = size / rc["font.size"]
        for key in _FONT_KEYS:
            rc[key] = rc[key] * scale
    if settings.get("grid") is False:
        rc["axes.grid"] = False
    elif settings.get("grid") is True:
        rc["axes.grid"] = True
        rc["axes.grid.axis"] = "both"
    return rc


@contextmanager
def context(settings: dict):
    with matplotlib.rc_context(rc_for(settings)):
        yield
