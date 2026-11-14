"""Survival curves on charts: the Kaplan-Meier estimate as steps.

The numbers are those of table.survival_over_time() (simplydata.table.survival.kaplan_meier,
the same as R survfit()): each curve starts at 1 at time 0, drops at each event time and
ends at the last time followed in its group, as R plot(survfit(...)). Censored rows are
small upright ticks on the curve at their time. confidence_band=True shades the 95 %
confidence interval (log scale, R's default conf.type = "log"). at_risk_table=True writes
the number still at risk (time at or after the tick) under each tick of the x-axis, one row
per group, as in papers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._errors import SimplydataError, friendly
from ..table.survival import check_time, group_order, is_empty, kaplan_meier, read_event
from ._helpers import check, check_thickness, check_transparency, colour_role, default_colour, opacity, palette_for
from .charts import BAND_SHADE, _apply_order, _remember

Y_LABEL = "Survival (share without the event)"
RISK_LINE = 11  # points between the rows of the at-risk table


def _rows_used(data, time, event, group):
    """The rows with time, event (and group) filled in, as plain lists."""
    columns = [c for c in (time, event, group) if c]
    empty = data[columns].map(is_empty).any(axis=1)
    return data.loc[~empty]


def _group_names(ctx, table, group):
    """Groups A to Z (as the table words), or in the order of order_legend() / a category column.
    Taken from the whole table, so colours stay the same on every panel."""
    whole = ctx.store.get("whole_table")
    whole = whole if whole is not None and group in whole.columns else table
    asked = (ctx.settings.get("order") or {}).get("colour")
    ordered = _apply_order(ctx, whole[[group]].dropna(), {"colour": group})
    if asked or isinstance(whole[group].dtype, pd.CategoricalDtype):
        return list(pd.unique(ordered[group]))
    return group_order(whole[group].dropna().tolist())


def _draw_survival(ctx, x, y, colour, confidence_band, censor_marks, transparency=None, thickness=None):
    ax = ctx.ax
    group, fixed = colour_role(ctx.data, colour)
    table = _rows_used(ctx.data, x, y, group)
    times = check_time(table[x].tolist(), x) if len(table) else np.array([])
    events = read_event(table[y].tolist(), y)[0] if len(table) else np.array([], dtype=bool)
    if group:
        names = _group_names(ctx, table, group)
        colours = palette_for(pd.DataFrame({group: names}), group, ctx.settings)
        keys = np.asarray(table[group].tolist(), dtype=object)
        parts = [(n, c, keys == n) for n, c in zip(names, colours)]
    else:
        parts = [(None, default_colour(ctx.settings, fixed), np.ones(len(times), dtype=bool))]
    risk = []
    for name, c, rows in parts:
        if not rows.any():
            continue
        km = kaplan_meier(times[rows], events[rows])
        end = km["end"]
        xs = np.concatenate([[0.0], km["time"], [end]])
        ys = np.concatenate([[1.0], km["survival"], [km["survival"][-1] if len(km["survival"]) else 1.0]])
        ax.step(xs, ys, where="post", color=c, label=None if name is None else str(name),
                alpha=opacity(transparency), linewidth=thickness)
        if confidence_band:
            low = np.concatenate([[1.0], km["low"], [km["low"][-1] if len(km["low"]) else 1.0]])
            high = np.concatenate([[1.0], km["high"], [km["high"][-1] if len(km["high"]) else 1.0]])
            ax.fill_between(xs, low, high, step="post", color=c, alpha=BAND_SHADE, linewidth=0)
        if censor_marks and len(km["censor_time"]):
            ax.plot(km["censor_time"], km["censor_survival"], linestyle="none", marker="|", markersize=8,
                    markeredgewidth=1.2, color=c, alpha=opacity(transparency))
        risk.append((name, c, times[rows]))
    ax.set_ylim(0, 1.03)
    ax.set_xlim(left=0)
    if ax.get_xlabel() == "":
        ax.set_xlabel(x)
    if ax.get_ylabel() == "":
        ax.set_ylabel(Y_LABEL)
    if group and risk:
        ax.legend(title=group)
    ctx.store.setdefault("at_risk", {})[id(ax)] = risk
    _remember(ctx, "survival_curves", table, {"x": x, "y": None, "colour": group})


def _draw_at_risk(ctx):
    """Numbers at risk under the x-axis ticks, one row per group (after ranges and ticks are set)."""
    ax = ctx.ax
    risk = ctx.store.get("at_risk", {}).get(id(ax))
    if not risk:
        return
    low, high = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if low - 1e-9 <= t <= high + 1e-9]
    size = ax.xaxis.get_label().get_fontsize() * 0.85
    gap = ax.xaxis.get_ticklabels()[0].get_fontsize() * 1.6 + 4 if ax.xaxis.get_ticklabels() else 16
    # the names stand left of the first column of numbers (half the widest number, plus a gap)
    digits = max(len(str(len(times))) for _, _, times in risk)
    left = -(digits * size * 0.3 + 6)
    ax.annotate("Number at risk", xy=(0, 0), xycoords="axes fraction", xytext=(left, -gap), textcoords="offset points",
                ha="right", va="top", fontsize=size, fontweight="bold")
    for row, (name, c, times) in enumerate(risk):
        offset = -gap - (row + 1) * RISK_LINE * size / 9
        if name is not None:
            ax.annotate(str(name), xy=(0, 0), xycoords="axes fraction", xytext=(left, offset),
                        textcoords="offset points", ha="right", va="top", fontsize=size, color=c)
        for t in ticks:
            ax.annotate(str(int((times >= t).sum())), xy=(t, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, offset), textcoords="offset points", ha="center", va="top", fontsize=size,
                        color=c)
    ax.xaxis.labelpad = (len(risk) + 1) * RISK_LINE * size / 9 + 6


class SurvivalChartWords:
    @friendly
    def survival_curves(self, time=None, event=None, colour=None, confidence_band=False, censor_marks=True,
                        at_risk_table=False, transparency=None, thickness=None):
        """The share still without the event over time (Kaplan-Meier curves).
        chart.survival_curves(time="days", event="died", colour="treatment") (with sdata.example("trial"))

        time= is the time until the event or the end of follow-up; event= says whether the
        event happened (yes/no, True/False or 1/0). colour="treatment" draws one curve per
        group, or colour="blue" one colour. Each curve starts at 1 and drops at each event;
        censored rows (followed without the event) are small ticks (censor_marks=False hides
        them). confidence_band=True shades the 95 % confidence interval. at_risk_table=True
        writes the number still at risk under the x-axis. transparency= (0 to 1) makes the curves
        see-through and thickness= sets their width in points. The numbers are those of
        table.survival_over_time() (as R survfit()).
        """
        example = 'chart.survival_curves(time="days", event="died", colour="treatment")'
        for option, value in (("time", time), ("event", event)):
            if not isinstance(value, str):
                raise SimplydataError(f".survival_curves() needs {option}= as a column name in quotes.\n"
                                      f"For example {example} (with sdata.example(\"trial\")).")
        if time == event:
            raise SimplydataError(f'"{time}" cannot be both time= and event=.\nFor example {example}.')
        for option, value in (("confidence_band", confidence_band), ("censor_marks", censor_marks),
                              ("at_risk_table", at_risk_table)):
            if not isinstance(value, bool):
                raise SimplydataError(f"{option}= in .survival_curves() should be True or False, not {value!r}.")
        transparency = check_transparency("survival_curves", transparency)
        thickness = check_thickness("survival_curves", thickness)
        group, _ = colour_role(self._data, colour)
        if group is not None and not isinstance(group, str):
            raise SimplydataError(f'.survival_curves() needs colour= as a column name or a colour.\nFor example {example}.')
        if group in (time, event):
            raise SimplydataError(f'"{group}" cannot be both colour= and time= or event=.')
        check(self._data, x=time, y=event, colour=group)
        rows = _rows_used(self._data, time, event, group)
        if not len(rows):
            raise SimplydataError(f'.survival_curves() found no rows with "{time}" and "{event}" filled in.')
        check_time(rows[time].tolist(), time)  # say what is wrong on the user's line
        read_event(rows[event].tolist(), event)
        self._need_axes("survival_curves")
        # x= and y= so the chart notes count the rows left out because of empty cells
        self._add("survival_curves", "draw", _draw_survival, x=time, y=event, colour=colour,
                  confidence_band=confidence_band, censor_marks=censor_marks, transparency=transparency,
                  thickness=thickness)
        if at_risk_table:
            self._replace("survival_at_risk", "finish", _draw_at_risk)
        return self
