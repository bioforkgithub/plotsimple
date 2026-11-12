"""The Table: one class that brings all the table word groups together."""

from __future__ import annotations

from .._errors import SimplydataError, friendly
from ._base import BaseTable
from .columns import ColumnWords
from .rows import RowWords
from .stat_tests import StatTestWords
from .summaries import SummaryWords


class Table(RowWords, ColumnWords, SummaryWords, StatTestWords, BaseTable):
    """A table. Make one with sdata.table(...), then add steps: table.keep_rows(...), table.sort_by(...)."""


@friendly(renamed={"big_data": "fast"})
def table(data=None, sheet=None, fast: bool = False) -> Table:
    """Start a table.

    sdata.table(sdata.example("marks"))         a practice table
    sdata.table("my_file.xlsx")              a file (Excel, CSV, TSV, Parquet, JSON)
    sdata.table("my_file.xlsx", sheet="2026") one sheet of an Excel file
    fast=True                              use the fast engine (Polars), for big data
    """
    if not isinstance(fast, bool):
        raise SimplydataError("fast= should be True or False.")
    if data is None:
        raise SimplydataError('sdata.table() needs some data, for example sdata.table(sdata.example("marks")).')
    return Table(data, sheet=sheet, fast=fast)
