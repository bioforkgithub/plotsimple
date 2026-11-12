# plotsimple

**Plain-English plotting and data for everyone.** One small vocabulary you can learn in a day,
built on matplotlib, seaborn, pandas, Polars and NumPy, so you don't have to learn them first.

> Status: early development (v0.1 in progress). The word list may still change.

```python
import plotsimple as ps

chart = ps.chart("marks.xlsx")
chart.bars(x="name", y="marks")
chart.title_top("My class marks")
chart.label_y("Marks")
chart.save("marks.png")
```

## Install

```
pip install plotsimple            # coming soon
pip install plotsimple[fast]      # adds the fast engine for big data
```

## Ideas behind it

- **Talk to the program.** Each step reads like a short sentence: `chart.title_top(...)`,
  `chart.label_x(...)`, `chart.line_at_y(50)`.
- **What + where.** A few place words repeat everywhere: `top`, `bottom`, `x`, `y`, `at`, `between`.
- **Order doesn't matter.** Write the steps in any order; plotsimple draws them correctly.
- **Friendly errors.** Misspell a column and plotsimple tells you what you probably meant.

The full design is in [PLAN.md](PLAN.md).

## Licence

MIT. Contributions are welcome: open an issue to suggest a word or report a problem.
