# plotsimple

**Plain-English plotting and data for everyone.** One small vocabulary you can learn in a day,
built on matplotlib, seaborn, pandas, Polars, NumPy and SciPy (with narwhals linking pandas and
Polars), so you don't have to learn them first.

> Status: early development (charts, tables, numbers and statistical tests ready). The word list may still change: an old word keeps working for one release and prints a one-line note with its new name.

```python
import plotsimple as ps

story = ps.start(example="marks")       # a made-up table; your own: ps.start(file="marks.xlsx")
story.keep_rows("marks", above=50)
story.average_of("marks", for_each="class")   # says the answer in a sentence
story.draw_bars(x="class", y="marks")
story.title_top("Good marks in each class")
story.label_y("Marks")
story.save_chart("marks.png")
story.save_table("good_marks.xlsx")
```

A **story** is one object for the whole analysis: every table word and every chart word works
on it, chart types start with `draw_`, and `story.undo()` and `story.steps()` cover table and
chart steps together. A chart uses the table as it was when it was first drawn.
`ps.table(...)` and `ps.chart(...)` are still there underneath, for programmers.

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
lesson 14 the statistical tests: is a difference real, or could it be chance? Lesson 15 tells a
whole story, from opening a file to a saved figure and table. Each lesson has an
example you can run, says what every line does, and ends with a small task to try yourself.

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
