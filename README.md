# plotsimple

**Plain-English plotting and data for everyone.** One small vocabulary you can learn in a day,
built on matplotlib, seaborn, pandas, Polars, NumPy and SciPy (with narwhals linking pandas and
Polars), so you don't have to learn them first.

> Status: early development (charts, tables, numbers and statistical tests ready). The word list may still change: an old word keeps working for one release and prints a one-line note with its new name.

```python
import plotsimple as ps

table = ps.table(ps.example("marks"))   # a made-up table; your own: ps.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.average_of("marks", for_each="class")   # says the answer in a sentence
table.save("good_marks.xlsx")

chart = table.chart()                   # or ps.chart(table)
chart.bars(x="class", y="marks", error="standard error")
chart.title_top("Good marks in each class")
chart.label_y("Marks")
chart.save("marks.png")
```

Tables, charts and numbers are separate objects, as in pandas, Polars, NumPy and matplotlib:
one cleaned table can feed many charts, and charts go side by side with
`ps.grid(a, b, letters=True)` (panels A, B, C ... as in papers).

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

There are fifteen short lessons. Lessons 1 to 8 teach charts, from your first chart to
animations. Lessons 9 to 12 teach tables: looking at your data, choosing rows and columns,
cleaning it, and summing it up. Lesson 13 teaches numbers (lists and grids of numbers), and
lesson 14 the statistical tests: is a difference real, or could it be chance? Lesson 15 puts it
all together, from a question and a file to a figure with panels A and B and a saved table. Each
lesson has an example you can run, says what every line does, explains the idea behind it
(average or median, standard deviation or standard error, p-values) and ends with a small task
to try yourself.

- `ps.lesson()` lists the lessons
- `ps.words()` lists every word, with an example (also in [docs/cheatsheet.md](docs/cheatsheet.md))
- `ps.help("bars")` explains one word: what it does, its options and an example
  (table and number words too: `ps.help("keep_rows")`, `ps.help("average")`)
- `ps.example()` lists the made-up tables to practise with (marks, weather, fruit, growth, island)

## Tables

Plain words for looking at data, cleaning it and summing it up, on pandas or Polars:

```python
table = ps.table(ps.example("marks"))      # or ps.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.sort_by("marks", biggest_first=True)
table.average_of("marks", for_each="class")  # says the answer in a sentence
table.compare_averages("marks", between="class")   # real difference, or chance?
table.undo()                               # take the last step back
table.save("good_marks.xlsx")
chart = table.chart()                      # draw it
```

## Numbers

Plain words over NumPy for lists and grids of numbers:

```python
nums = ps.numbers([12, 15, 9, 20, 14])
bigger = nums * 2                          # symbols make new numbers
nums.keep_numbers(above=10)                # words change the numbers you have ...
nums.undo()                                # ... and can be taken back
nums.average()
grid = ps.number_grid([[3, 5, 2], [8, 6, 4]])
grid.total(for_each="row")
```

## Ideas behind it

- **Talk to the program.** Each step reads like a short sentence: `chart.title_top(...)`,
  `chart.label_x(...)`, `chart.line_at_y(50)`.
- **What + where.** A few place words repeat everywhere: `top`, `bottom`, `x`, `y`, `at`, `between`.
- **Guess a word from its sibling.** `keep_rows` and `keep_first_rows`, `show_first_rows`;
  `average_of` and `show="average"`; `label_x` and `label_colour_bar`.
- **The word names the idea.** `error="standard error"`, `fit_line(confidence_band=True)`,
  `compare_averages` (and the result says "Welch's t-test" for when you write it up).
- **Order doesn't matter.** Write the steps in any order; plotsimple draws them correctly.
- **Friendly errors.** Misspell a column and plotsimple tells you what you probably meant.

The full design is in [PLAN.md](PLAN.md).

## Licence

MIT. Contributions are welcome: open an issue to suggest a word or report a problem.
