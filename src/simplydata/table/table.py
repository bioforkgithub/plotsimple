"""The Table: one class that brings all the table word groups together."""

from __future__ import annotations

from .._data import reading_options
from .._errors import SimplydataError, friendly
from ._base import BaseTable
from .between import BetweenWords
from .columns import ColumnWords
from .distribution_words import DistributionTableWords
from .fitting import FittingWords
from .models import ModelWords
from .patterns import PatternWords
from .positions import PositionWords
from .rows import RowWords
from .stat_tests import StatTestWords
from .summaries import SummaryWords
from .survival import SurvivalWords


class Table(PositionWords, RowWords, ColumnWords, SummaryWords, StatTestWords, FittingWords, ModelWords, PatternWords, SurvivalWords,
            BetweenWords, DistributionTableWords, BaseTable):
    """A table. Make one with sdata.table(...), then add steps: table.keep_rows(...), table.sort_by(...)."""


@friendly(renamed={"big_data": "fast"})
def table(data=None, sheet=None, fast: bool = False, separator=None, decimal_mark=None, thousands_mark=None,
          empty_values=None, skip_rows=None, encoding=None, table_number=None) -> Table:
    """Start a table.

    sdata.table(sdata.example("marks"))         a practice table
    sdata.table("my_file.xlsx")              a file (Excel, CSV, TSV, Parquet, JSON; Word, SPSS, Stata)
    sdata.table("my_file.xlsx", sheet="2026") one sheet of an Excel file; sheet="all" stacks every sheet
    sdata.table("data/*.csv")                several files stacked, with a "file" column (or a folder: "data/")
    sdata.table("https://example.org/data.csv") a file on the web
    fast=True                              use the fast engine (Polars), for big data

    Reading options, for files whose values are written differently:
    separator=";"            the character between values ("tab" for tabs)
    decimal_mark=","         3,5 means three and a half
    thousands_mark="."       1.000 means one thousand
    empty_values=["NA", "-"] markers that mean "no value"
    skip_rows=2              skip lines above the column names
    encoding="latin-1"       for text files not saved as UTF-8
    table_number=2           which table in a Word file ("all" stacks them)
    """
    if not isinstance(fast, bool):
        raise SimplydataError("fast= should be True or False.")
    if data is None:
        raise SimplydataError('sdata.table() needs some data, for example sdata.table(sdata.example("marks")).')
    reading = reading_options(separator=separator, decimal_mark=decimal_mark, thousands_mark=thousands_mark,
                              empty_values=empty_values, skip_rows=skip_rows, encoding=encoding,
                              table_number=table_number)
    return Table(data, sheet=sheet, fast=fast, _reading=reading)
