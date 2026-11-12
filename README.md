# plotsimple

**Plain-English plotting and data for everyone.** One small vocabulary you can learn in a day,
built on matplotlib, seaborn, pandas, Polars, NumPy and SciPy (with narwhals linking pandas and
Polars), so you don't have to learn them first.

> Status: early development (charts and tables ready; numbers and statistical tests in progress). The word list may still change.

```python
import plotsimple as ps

marks = ps.example("marks")        # a made-up table to practise with
chart = ps.chart(marks)            # your own file works too: ps.chart("marks.xlsx")
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

## Learn it in a day

plotsimple teaches itself. Open Python and type:

```python
import plotsimple as ps
ps.lesson(1)
```

There are twelve short lessons. Lessons 1 to 8 teach charts, from your first chart to
animations. Lessons 9 to 12 teach tables: looking at your data, choosing rows and columns,
cleaning it, and summing it up. Each lesson has an example you can run, says what every line
does, and ends with a small task to try yourself.

- `ps.lesson()` lists the lessons
- `ps.words()` lists every word, with an example (also in [docs/cheatsheet.md](docs/cheatsheet.md))
- `ps.help("bars")` explains one word: what it does, its options and an example
  (table words too: `ps.help("keep_rows")`)
- `ps.example()` lists the made-up tables to practise with (marks, weather, fruit, growth, island)

## Tables

Plain words for looking at data, cleaning it and summing it up, on pandas or Polars:

```python
table = ps.table(ps.example("marks"))      # or ps.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.sort_by("marks", biggest_first=True)
print(table.average_of("marks", for_each="class"))
table.undo()                               # take the last step back
table.save("good_marks.xlsx")
chart = table.chart()                      # draw it
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
