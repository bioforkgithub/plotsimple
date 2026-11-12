"""simplydata: plain-English plotting and data for everyone.

    import simplydata as sdata

    table = sdata.table(sdata.example("marks"))     # or sdata.table("results.xlsx")
    table.keep_rows("marks", above=50)
    chart = table.chart()                       # or sdata.chart(table)
    chart.bars(x="class", y="marks")
    chart.title_top("Good marks")
    chart.save("marks.png")

Numbers too: nums = sdata.numbers([3, 5, 2, 8]), then nums.average().

New here? Type sdata.lesson(1). Every word: sdata.words(). One word: sdata.help("bars").
"""

from ._errors import SimplydataError
from .chart import Chart, chart
from .data import example
from .table import Table, table
from .numbers import NumberGrid, Numbers, number_grid, numbers, random_numbers
from .words.layout import grid, one_above_other, side_by_side
from .words.stats import pairs

# Last, on purpose: the function sdata.words must win over the subpackage simplydata.words
# (sdata.table and sdata.numbers win over their subpackages the same way, by being imported after them).
from .learn import help, lesson, words  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "Chart", "NumberGrid", "Numbers", "SimplydataError", "Table", "chart", "example", "grid", "help",
    "lesson", "number_grid", "numbers", "one_above_other", "pairs", "random_numbers", "side_by_side",
    "table", "words", "__version__",
]
