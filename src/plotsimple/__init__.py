"""plotsimple: plain-English plotting and data for everyone.

    import plotsimple as ps

    marks = ps.example("marks")
    chart = ps.chart(marks)
    chart.bars(x="name", y="marks")
    chart.title_top("My class marks")
    chart.save("marks.png")

Tables too: table = ps.table(marks), then table.keep_rows("marks", above=50).
Numbers too: nums = ps.numbers([3, 5, 2, 8]), then nums.average().

New here? Type ps.lesson(1). Every word: ps.words(). One word: ps.help("bars").
"""

from ._errors import PlotsimpleError
from .chart import Chart, chart
from .data import example
from .table import Table, table
from .numbers import NumberGrid, Numbers, number_grid, numbers, random_numbers
from .words.layout import grid, one_above_other, side_by_side
from .words.stats import pairs

# Last, on purpose: the function ps.words must win over the subpackage plotsimple.words
# (ps.table and ps.numbers win over their subpackages the same way, by being imported after them).
from .learn import help, lesson, words  # noqa: E402

__version__ = "0.1.0.dev0"

__all__ = [
    "Chart", "NumberGrid", "Numbers", "PlotsimpleError", "Table", "chart", "example", "grid", "help",
    "lesson", "number_grid", "numbers", "one_above_other", "pairs", "random_numbers", "side_by_side",
    "table", "words", "__version__",
]
