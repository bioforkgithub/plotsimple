"""The Chart: one class that brings all the word groups together."""

from __future__ import annotations

from ._core import BaseChart
from ._errors import PlotsimpleError
from .words.axes import AxesWords
from .words.charts import ChartWords
from .words.layout import LayoutWords
from .words.look import LookWords
from .words.stats import StatsWords
from .words.text import TextWords


class Chart(ChartWords, StatsWords, TextWords, AxesWords, LookWords, LayoutWords, BaseChart):
    """A chart. Make one with ps.chart(...), then add steps: chart.bars(...), chart.title_top(...)."""


def chart(data=None, sheet=None, big_data: bool = False) -> Chart:
    """Start a chart. ps.chart(ps.example("marks"))

    ps.chart(table)                        a pandas or Polars table, like ps.example("marks")
    ps.chart("my_file.xlsx")               your own file (Excel, CSV, TSV, Parquet, JSON)
    ps.chart("my_file.xlsx", sheet="2026") one sheet of an Excel file
    ps.chart()                             no table: give lists to each step instead
    big_data=True                          always use the fast engine for big files
    """
    if not isinstance(big_data, bool):
        raise PlotsimpleError("big_data= should be True or False.")
    return Chart(data, sheet=sheet, big_data=big_data)
