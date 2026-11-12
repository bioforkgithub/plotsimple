"""plotsimple: plain-English plotting and data for everyone.

    import plotsimple as ps

    story = ps.start(example="marks")     # or ps.start(file="results.xlsx")
    story.keep_rows("marks", above=50)
    story.draw_bars(x="class", y="marks")
    story.title_top("Good marks")
    story.save_chart("marks.png")

Underneath, for programmers: ps.chart(...) and ps.table(...) on their own.
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
from .story import Story, start

# Last, on purpose: the function ps.words must win over the subpackage plotsimple.words
# (ps.table and ps.numbers win over their subpackages the same way, by being imported after them).
from .learn import help, lesson, words  # noqa: E402

__version__ = "0.1.0.dev0"

__all__ = [
    "Chart", "NumberGrid", "Numbers", "PlotsimpleError", "Table", "chart", "example", "grid", "help",
    "lesson", "number_grid", "numbers", "one_above_other", "pairs", "random_numbers", "side_by_side",
    "start", "Story", "table", "words", "__version__",
]
