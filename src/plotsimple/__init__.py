"""plotsimple: plain-English plotting and data for everyone.

    import plotsimple as ps

    marks = ps.example("marks")
    chart = ps.chart(marks)
    chart.bars(x="name", y="marks")
    chart.title_top("My class marks")
    chart.save("marks.png")

New here? Type ps.lesson(1). Every word: ps.words(). One word: ps.help("bars").
"""

from ._errors import PlotsimpleError
from .chart import Chart, chart
from .data import example
from .words.layout import grid, one_above_other, side_by_side
from .words.stats import pairs

# Last, on purpose: the function ps.words must win over the subpackage plotsimple.words.
from .learn import help, lesson, words  # noqa: E402

__version__ = "0.1.0.dev0"

__all__ = [
    "Chart", "PlotsimpleError", "chart", "example", "grid", "help", "lesson", "one_above_other",
    "pairs", "side_by_side", "words", "__version__",
]
