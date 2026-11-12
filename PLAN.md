# plotsimple — design plan

*Status: design stage, no code yet. Last updated 2026-11-12.*

## 1. The idea

plotsimple is **a dictionary, not a new engine**. One small, plain-English vocabulary
that translates into the right calls in **matplotlib, seaborn, plotnine, pandas and NumPy**
(plus SciPy for statistics). Learn plotsimple, and you never have to learn the five
libraries underneath.

Like a phrasebook that lets you speak Spanish, Italian and French without learning each
grammar.

- The libraries underneath do all drawing and calculation.
- plotsimple's job is the **translation**: each word must map to the right call with the
  right settings. That is what we test.
- Users never see the libraries underneath. There is **no bridge** to raw matplotlib
  (no `show_code()`, no `.matplotlib()`).

## 2. Goals

1. **Simple words, full power.** Every word is simple enough for a child to read, yet together
   they unlock the full power of the libraries underneath, for beginners and researchers alike.
2. **Learn everything in one day**, write programs the next day.
3. **The hardest plots are as easy as the simplest** (multi-panel figures, second
   y-axis, zoom inset, notes with arrows, small multiples, animation, polar, 3D).
4. **Better than seaborn and plotnine** for learners, while covering what they do.
5. Later: replace everyday use of pandas and NumPy as well (tables first, then arrays).

## 3. Basics

| Topic | Decision |
|---|---|
| Name | `plotsimple`, used as `import plotsimple as ps` (name free on PyPI on 2026-11-12) |
| Licence | MIT, open on GitHub, contributions welcome |
| Python | 3.10 and newer |
| Audience | Everyone: beginners, students, teachers and researchers. "Simple enough for a school child to read" is the test for every word, not a limit on what it can do |
| Data | Plain lists, Excel/CSV files, pandas tables, Polars tables; results stay plain pandas/NumPy (or Polars, if Polars went in) |
| Speed | Automatic: Polars tables stay Polars, files over 100 MB use Polars, everything else pandas. Override in plain words: `fast=True`. A one-line message says when fast mode is used. |
| Style | One step per line on a chart; chaining also allowed |
| Look | Clean paper style by default (white, large text, colour-blind-safe colours) |
| Showing | Charts appear automatically in Jupyter/VS Code/Spyder; `chart.show()` in scripts |
| Saving | `chart.save("x.png")` or `chart.save("x.pdf")`, format from the file name |
| Spelling | Both `colour` and `color`, `grey` and `gray` |
| Errors | Friendly message, "did you mean …?", list of columns, link to request a missing word on GitHub |
| Learning | Built-in lessons `ps.lesson(1)` … `ps.lesson(15)`, `ps.help("bars")`, `ps.words()` |
| Renaming | A renamed word or option keeps working for one release and prints one line, once per session: `Note: "remove_empty_rows" is now called "remove_rows_with_empty_cells".` Old names are never listed in `ps.words()`, the cheat sheet or the lessons; `ps.help("old_name")` points to the new one |

## 4. Grammar rules

1. **What + where**: `title_top`, `caption_bottom`, `legend_bottom`, `line_at_y`.
2. **Axes are `x` and `y`** (the words children learn in school maths).
3. Anything that belongs to one axis **ends in `_x` or `_y`**: `label_y`, `range_y`,
   `log_scale_y`, `ticks_x`.
4. **Mathematical terms keep their names**: log, histogram, density, polar.
5. Other jargon becomes plain words: `dots` (scatter), `sideways=True` (horizontal bars).
6. Calling two chart types on the same chart puts them on top of each other.

### Six patterns for the whole vocabulary

A user who knows one word of a pattern should be able to guess the others.

| # | Pattern | Examples |
|---|---|---|
| 1 | **verb + thing** changes a table or numbers: `keep_`, `remove_`, `add_`, `make_`, `fill_`, `turn_`, `rename_`, `sort_` + rows / columns / cells / numbers | `keep_rows`, `remove_columns`, `fill_empty_cells`, `remove_rows_with_empty_cells`, `fill_empty_numbers` |
| 2 | **`draw_` + chart type** draws on a story (a plain noun on a `chart`) | `draw_bars`, `draw_histogram`, `draw_all_dots` (`bars`, `histogram` on a chart) |
| 3 | **thing + where** places something on a chart; where = `top`/`bottom`/`left`/`right`/`x`/`y`/`at`/`between` | `title_top`, `label_y`, `label_colour_bar`, `line_at_y`, `legend_bottom`, `zoom_box(where="top right")` |
| 4 | **measure + `_of`(column, `for_each=`)** for one column; **relation + `_between`(a, b)** for two; **`compare_` + measure** for tests | `median_of`, `correlation_between`, `compare_averages` |
| 5 | **`show_` / `hide_` + thing** to look at something without changing it | `show_first_rows`, `show_grid`/`hide_grid`, `show_legend`/`hide_legend` |
| 6 | **`as_` + library** hands the data to another Python tool; **`save_` + thing** writes a file | `as_pandas`, `as_polars`, `as_numpy`; `save_table`, `save_chart` |

**One option name, one meaning, everywhere:** `for_each=` (groups), `colour=` (one colour, or a
colour per group), `between=[low, high]` and `x_between=`/`y_between=` (every span),
`from_column=`, `with_value=`, `biggest_first=`, `per_row=` (charts in a row), `columns=` (table
columns only), `label=` (text on a line or band), `fast=`, `drawn_as=`. Measure names are the same
wherever they appear: `"average"`, `"median"`, `"total"`, `"count"`, `"smallest"`, `"biggest"`,
`"standard deviation"`, `"standard error"`, `"confidence interval"` (in `show=`, `error=`,
`error_bars` and the `_of` words). Summary words print their answer as one sentence, so a story
needs no `print()`.

**The word names the idea.** The word says the question in plain words; help and the printed
result name the technical term ("Welch's t-test", "linear regression"), so a learner can follow
a paper's Methods section.

**Checklist for a new word:** can a user who has never seen it guess it from one sibling? If
not, rename it before it is released.

### The Story: one object for a whole analysis

`ps.start(...)` opens a table and returns a **Story**: the table (with its history) and the
charts drawn from it. It is the front door that the lessons and the README teach; `ps.table`
and `ps.chart` stay underneath as the layer for programmers.

```python
story = ps.start(file="results.xlsx")   # or example="marks", table=df, sheet=, fast=
story.remove_rows_with_empty_cells()
story.keep_rows("score", above=50)
story.draw_bars(x="class", y="score")
story.title_top("Good scores")
story.save_chart("results.png")
story.save_table("clean.xlsx")
```

- **Every table word works on a story** and changes its table. A summary or test that gives
  back a table gives back a **new story**, so `totals.draw_bars(...)` works; plain numbers stay
  numbers.
- **Chart types get `draw_`** (pattern 2), so every line starts with a verb. Every other chart
  word (`title_top`, `label_y`, `fit_line`, `style`, `size` …) changes the **current chart**:
  the one drawn last. A chart word before any `draw_` says "Draw a chart first".
- **Snapshot rule:** a chart uses the table as it was when its first `draw_` step was written.
  Later table steps do not change it, so the program reads top to bottom. `new_chart()` starts
  another chart, which uses the table as it is when it is drawn.
- **One history:** `undo()` takes back the last step, table or chart; `steps()` lists both as
  sentences ("3. draw bars of score for each class"); `start_again()` goes back to the opened
  table with no charts (and can be undone).
- **Several charts:** `put_charts_together(per_row=2, letters=True)` puts the story's charts on
  one page with panel letters A, B, C …; then `save_chart` saves the page, and `title_top`,
  `style` and `size` change the page. Charts from different tables: `ps.grid(story_a, story_b,
  letters=True)`, which takes each story's current chart.
- **One meaning per word:** `save_table` / `save_chart` (plain `save` picks by file type, since
  table and picture types never overlap); `show_chart` (a bare `show` would be ambiguous next to
  `show_first_rows`); `table.chart()` is not needed on a story.

```python
import plotsimple as ps

chart = ps.chart("marks.xlsx")
chart.bars(x="name", y="marks")
chart.title_top("My class marks")
chart.label_x("Student")
chart.label_y("Marks")
chart.line_at_y(50)                  # pass mark
chart.shade_between_y(40, 60)
chart.legend_bottom()
chart.save("marks.png")
```

## 5. Word list (draft v0.1)

### 5.1 Start, show, save
| Word | Example |
|---|---|
| `ps.chart` | `chart = ps.chart("marks.xlsx")` · `ps.chart(table)` · `ps.chart()` for lists |
| `show` | `chart.show()` |
| `save` | `chart.save("marks.png")` · `chart.save("marks.pdf")` |

### 5.2 Chart types
| Word | Example |
|---|---|
| `bars` | `chart.bars(x="name", y="marks")` · `stacked=True` · `sideways=True` |
| `lines` | `chart.lines(x="day", y="temp", colour="city")` |
| `dots` | `chart.dots(x="height", y="weight", size="age", shape="type")` |
| `area` | `chart.area(x="year", y="sales")` |
| `histogram` | `chart.histogram(x="marks", bins=10)` |
| `boxes` | `chart.boxes(x="class", y="marks")` |
| `violins` | `chart.violins(x="class", y="marks")` |
| `heatmap` | `chart.heatmap(numbers=True)` · `cluster=True` |
| `pie` | `chart.pie(labels="fruit", values="count")` |
| `polar` | `chart.polar(angle="direction", distance="speed")` |
| `contour` | `chart.contour(x="lon", y="lat", z="depth")` |
| `surface` | `chart.surface(x="lon", y="lat", z="depth")` |
| `dots_3d` | `chart.dots_3d(x="a", y="b", z="c")` |

### 5.3 Statistics
| Word | Example |
|---|---|
| `show=` | `chart.bars(x="class", y="marks", show="average")` · `"median"` · `"total"` · `"count"` · `"smallest"` · `"biggest"` · `"standard deviation"` |
| `error=` | `error="standard deviation"` · `"standard error"` · `"confidence interval"` (`"sd"`, `"se"`, `"ci"` also work) |
| `error_bars` | `chart.error_bars("standard error")` · `chart.error_bars(column="error")` |
| `fit_line` | `chart.fit_line()` · `confidence_band=True` · `show_equation=True` |
| `fit_curve` | `chart.fit_curve(degree=2)` |
| `smooth_line` | `chart.smooth_line()` |
| `density` | `chart.density(x="marks", colour="class")` |
| `count_bars` | `chart.count_bars(x="fruit")` |
| `all_dots` | `chart.all_dots(x="class", y="marks")` |
| `side_histograms` | `chart.side_histograms()` |
| `ps.pairs` | `ps.pairs(table)` |

### 5.4 Lines, shading, words
| Word | Example |
|---|---|
| `line_at_x` / `line_at_y` | `chart.line_at_y(50)` |
| `shade_between_x` / `shade_between_y` | `chart.shade_between_y(40, 60)` |
| `title_top` | `chart.title_top("My class marks")` |
| `caption_bottom` | `chart.caption_bottom("Data from 2026")` |
| `label_x` / `label_y` | `chart.label_y("Marks")` |
| `note` | `chart.note("Highest!", x=5, y=98, arrow=True)` |
| `numbers_on_bars` | `chart.numbers_on_bars()` |
| `font_size` | `chart.font_size(14)` |

### 5.5 Axes
| Word | Example |
|---|---|
| `range_x` / `range_y` | `chart.range_y(0, 100)` |
| `log_scale_x` / `log_scale_y` | `chart.log_scale_y()` |
| `ticks_x` / `ticks_y` | `chart.ticks_x(0, 10, 20, 30)` |
| `rotate_labels_x` | `chart.rotate_labels_x(45)` |
| `second_y` | `chart.second_y("rain", drawn_as="bars", label="Rain (mm)")` |
| `flip_y` | `chart.flip_y()` |

### 5.6 Colours and look
| Word | Example |
|---|---|
| `colour` / `color` | `chart.colour("blue")` · inside a chart: `colour="site"` |
| `colours` | `chart.colours("blue", "orange", "green")` |
| `style` | `chart.style("paper")` · `"school"` · `"dark"` · `"presentation"` |
| `size` | `chart.size(width=8, height=6)` |
| `show_grid` / `hide_grid` | `chart.hide_grid()` |
| `legend_top` / `_bottom` / `_left` / `_right` | `chart.legend_bottom()` |
| `hide_legend` / `show_legend` | `chart.hide_legend()` |
| `label_colour_bar` | `chart.label_colour_bar("Depth (m)")` |

### 5.7 Several charts and hard tricks
| Word | Example |
|---|---|
| `ps.side_by_side` | `ps.side_by_side(a, b).save("both.png")` |
| `ps.one_above_other` | `ps.one_above_other(a, b)` |
| `ps.grid` | `ps.grid(a, b, c, d, per_row=2, same_y=True)` |
| `one_chart_per` | `chart.one_chart_per("site", per_row=3)` · `one_chart_per(down="site", across="year")` |
| `zoom_box` | `chart.zoom_box(x_between=[2, 4], y_between=[10, 20], where="top right")` |
| `animate` | `chart.animate(over="year")` then `chart.save("growth.gif")` |

### 5.8 Help and learning
| Word | Example |
|---|---|
| `ps.lesson` | `ps.lesson(1)` … `ps.lesson(14)` |
| `ps.help` | `ps.help("bars")` |
| `ps.words` | `ps.words()` prints the whole list |

## 6. Libraries underneath

| Library | Used for |
|---|---|
| matplotlib | Drawing, axes, layout, saving, animation |
| seaborn | Statistical charts (averages with error bars, density, violins, pairs, clustered heatmap) |
| plotnine | Grammar-style layered charts and two-way panels where it is the best fit |
| pandas | Reading files, tables, grouping (default for normal-sized data) |
| Polars | Fast engine for large data and Polars input; charts convert it to pandas before drawing |
| narwhals | Candidate helper so the same table code runs on both pandas and Polars |
| NumPy | Arrays and arithmetic |
| SciPy | t-based confidence intervals, curve fitting, KDE, clustering |

Design rule: each plotsimple word translates to **one** library. seaborn draws on matplotlib
charts, so those two mix freely; plotnine builds its own charts, so words that use it must
not be mixed with matplotlib/seaborn words on the same chart. Which library each word uses
is decided per word.

## 7. Our responsibility

The libraries do the calculations. **We are responsible for correct translation**: choosing
the right function and settings for each word. Known traps:

- Standard deviation: NumPy defaults to population SD (n), pandas to sample SD (n−1).
  plotsimple default: sample SD (n−1).
- Missing values: NumPy returns `nan`, pandas skips them. plotsimple must choose and say so.
- Confidence intervals: seaborn bootstraps, a t-interval differs slightly. Default to be
  decided; the method is always stated.

Testing: every word gets a test that its output matches the library underneath, and
statistics are checked against SciPy, seaborn and R on reference data.

## 8. Releases

| Version | Content |
|---|---|
| v0.1 | Charts (sections 5.1–5.8), lessons 1–8, friendly errors |
| v0.1 (also) | Accept Polars tables as chart input |
| v0.2 | Tables: plain-word layer over pandas and Polars (read, clean, keep, group, summarise), with automatic engine choice and `fast=True` |
| v0.3 | Arrays: plain-word layer over NumPy; statistical tests in plain words (comparing averages, comparing counts, fitting formulas) over SciPy |
| v0.4 | More statistics: regression with several predictors, mixed models, survival curves, multiple-testing correction, power analysis, PCA and ordination |
| v0.5 | Curve fitting and optimisation: fit any formula, find minima and maxima, solve equations |
| v0.6 | Signals and time series: smoothing, filtering, peaks, Fourier transforms, rolling averages, resampling dates |
| v0.7 | Maths: areas under curves, differential equations, matrices and linear algebra |

