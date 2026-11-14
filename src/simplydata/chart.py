"""The Chart: one class that brings all the word groups together."""

from __future__ import annotations

from ._core import BaseChart
from ._errors import SimplydataError, friendly
from .words.axes import AxesWords
from .words.charts import ChartWords
from .words.curves import CurveWords
from .words.layout import LayoutWords
from .words.look import LookWords
from .words.stats import StatsWords
from .words.survival import SurvivalChartWords
from .words.text import TextWords


class Chart(ChartWords, StatsWords, CurveWords, SurvivalChartWords, TextWords, AxesWords, LookWords, LayoutWords,
            BaseChart):
    """A chart. Make one with sdata.chart(...), then add steps: chart.bars(...), chart.title_top(...)."""


@friendly(renamed={"big_data": "fast"})
def chart(data=None, sheet=None, fast: bool = False) -> Chart:
    """Start a chart. sdata.chart(sdata.example("marks"))

    sdata.chart(table)                        a pandas or Polars table, like sdata.example("marks")
    sdata.chart("my_file.xlsx")               your own file (Excel, CSV, TSV, Parquet, JSON)
    sdata.chart("my_file.xlsx", sheet="2026") one sheet of an Excel file
    sdata.chart()                             no table: give lists to each step instead
    fast=True                              always use the fast engine (Polars) to read the file
    """
    if not isinstance(fast, bool):
        raise SimplydataError("fast= should be True or False.")
    return Chart(data, sheet=sheet, fast=fast)
