"""plotsimple: plain-English plotting and data for everyone.

    import plotsimple as ps

    chart = ps.chart("marks.xlsx")
    chart.bars(x="name", y="marks")
    chart.title_top("My class marks")
    chart.save("marks.png")
"""

from ._errors import PlotsimpleError
from .chart import Chart, chart

__version__ = "0.1.0.dev0"

__all__ = ["Chart", "PlotsimpleError", "chart", "__version__"]
