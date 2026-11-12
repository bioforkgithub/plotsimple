"""The Table: one class that brings all the table word groups together."""

from __future__ import annotations

from .._errors import PlotsimpleError
from ._base import BaseTable
from .columns import ColumnWords
from .rows import RowWords
from .summaries import SummaryWords


class Table(RowWords, ColumnWords, SummaryWords, BaseTable):
    """A table. Make one with ps.table(...), then add steps: table.keep_rows(...), table.sort_by(...)."""


def table(data=None, sheet=None, big_data: bool = False) -> Table:
    """Start a table.

    ps.table(ps.example("marks"))         a practice table
    ps.table("my_file.xlsx")              a file (Excel, CSV, TSV, Parquet, JSON)
    ps.table("my_file.xlsx", sheet="2026") one sheet of an Excel file
    big_data=True                          use the fast engine (Polars) for big data
    """
    if not isinstance(big_data, bool):
        raise PlotsimpleError("big_data= should be True or False.")
    if data is None:
        raise PlotsimpleError('ps.table() needs some data, for example ps.table(ps.example("marks")).')
    return Table(data, sheet=sheet, big_data=big_data)
