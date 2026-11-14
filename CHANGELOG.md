# Changes

Each release lists what was added, changed or fixed. When a word is renamed, the old word keeps
working for one release and prints a one-line note with its new name.

## Unreleased

More statistics, each checked against R.

### Fitting

- `table.fit_line(x=, y=)` is the main name of the straight-line fit, the same word as
  `chart.fit_line`; `fit_straight_line` still works for one release, with a note.
- `table.fit_curve(x=, y=, degree=2)` fits a polynomial by least squares (R
  `lm(y ~ poly(x, 2, raw = TRUE))`), with an interval for each term; `weights=` and `for_each=`.
- `table.fit_formula` takes `between=` (limits for the values to find, as R
  `nls(algorithm = "port")`), `weights=`, `for_each=`, `search="whole range"` (differential
  evolution, seed 1, then a least-squares polish) and `method=`, and gives standard errors,
  t, p-values, 95 % intervals, the residual standard error and AIC, as R `nls()`.
- `table.add_predicted_column("population = a * exp(b * (year - 2000))")` fits a formula and adds
  its value for every row (or takes the values of a fit with `using=`).
- On charts: `chart.fit_formula(...)` draws the fitted formula for each colour group and
  `chart.curve(...)` draws any formula of x, or a distribution.

### Finding and solving

- `sdata.find_smallest` and `sdata.find_biggest` find where a formula is smallest or biggest:
  on a range (Brent's method, R `optimize()`), from a start (Nelder-Mead, R `optim()`), with limits
  (L-BFGS-B), under rules (SLSQP; linear programming with HiGHS when everything is a straight line,
  with the value of one more of each rule; `whole_numbers=True` for mixed-integer programming), or
  over the whole range (differential evolution). The method is always printed, never switched
  silently.
- `sdata.solve` finds every solution of an equation in a range (R `uniroot()` on each sign change),
  all roots of a polynomial, complex ones too (R `polyroot()`), and solves sets of equations
  (R `nleqslv`), exactly when they are linear.
- `known=` takes the values of a `fit_formula` result, so a fitted curve can be solved or
  searched straight away.

### Values between points

- `table.values_at(column, at={"year": [2001, 2003]})` estimates a column between the known
  points: `joined_by="straight lines"` (R `approx()`), `"smooth curve"` (natural cubic spline),
  `"curve without overshoot"` (PCHIP), `"steps"` or `"nearest"`, or `smoothness="auto"` for a
  smoothing spline (R `smooth.spline()`); in two or more directions (bilinear on a full grid,
  triangles or a thin-plate spline for scattered points). `outside=` says what happens beyond the
  known points.
- `table.map_of(column, across=["east", "north"])` fills an evenly spaced map, ready for a contour
  map or heatmap.
- `fill_empty_cells(with_the="curve between neighbours")` and `"curve without overshoot"`;
  `chart.lines(joined_by=...)` and `chart.smooth_line(method="spline")`.

### Distributions

- `sdata.distribution("normal", average=50, standard_deviation=10)` for 26 named distributions
  (R's parameters with plain names; R's own names work too) and any other SciPy distribution.
  Its words: `chance` (above=, below=, at_most=, at_least=, between=, or exactly a count),
  `density_at`, `percentile`, `cut_off`, `middle_range`, `average`, `median`,
  `standard_deviation`, `variance`, `summary`, `random_numbers`, `chart`, `table` and
  `parameters`. Each sentence prints the R call that gives the same answer.
- `table.fit_distribution` and `nums.fit_distribution` fit by maximum likelihood (as R
  `MASS::fitdistr`) and give back the fitted distribution; `table.compare_distributions` ranks
  several by AIC, with the goodness-of-fit statistics of R `fitdistrplus::gofstat()`.
- `table.compare_shape_to(column, "gamma")` checks any smooth shape: the Anderson-Darling test with
  a Monte Carlo p-value that allows for the fitting (seed 1). `"normal"` keeps Shapiro-Wilk.
- On charts: `chart.fit_distribution("gamma")` over a histogram or density, and
  `chart.qq_dots(column, distribution="gamma")`.

### Names from other tools

- Names that people bring from pandas, NumPy, SciPy, R, matplotlib and ggplot2 (`sdata.minimize`,
  `sdata.curve_fit`, `sdata.pnorm`, `table.groupby`, `table.merge`, `chart.hist`, `chart.axhline`
  and about 400 more) never run: each says "In simplydata this is ..." with the word to use.
  `sdata.help("minimize")` explains them too.

### Dictionary

- `docs/manual/simplydataDictionary.pdf`: every word A to Z (one entry per word, with a part for
  each object that has it, such as `fit_line` on tables and charts), each with its meaning, a short
  example, its main options, the names other tools use for it, related words and the page of the
  manual that explains it (a link that opens the manual there), plus the lesson that teaches it.
  Its last part, "Coming from another tool?", turns every name from other tools and every old name
  into the simplydata word. It is made from the code by `docs/make_dictionary.py`; building the
  manual (`python docs/make_manual.py`) builds the dictionary too, with the manual's page numbers,
  and a test fails if a word has no page or the pages come from an older manual.

### Not in this release

Robust (outlier-resistant) curve fitting, gradients given by hand, profile-likelihood intervals
for formula fits, mixtures and truncated distributions, distributions of several variables, and
slopes and areas under splines.

### Models

- `table.explain(column, using=[...])` explains one column from several others: a linear model
  by least squares (multiple regression), with `"a * b"` for an interaction and text columns as
  groups. `outcome="yes or no"` is logistic regression, `outcome="counts"` Poisson regression
  (`"counts (overdispersed)"` quasi-Poisson), `grouped_by=` a mixed model with a random intercept
  per group, and `empty_cells="multiple imputation"` fills empty cells 20 times and pools the
  answers with Rubin's rules.
- `table.add_predicted_column` adds what the model expects for each row.

### Patterns across columns

- `add_principal_components` and `principal_components_of` (PCA), `add_clusters` (k-means or
  hierarchical), `distances_between_rows` (Bray-Curtis, Euclidean, Jaccard, Manhattan),
  `add_nmds` (non-metric multidimensional scaling) and `compare_profiles` (PERMANOVA, with a
  check of the spread of each group).

### Survival

- `survival_over_time` (Kaplan-Meier), `compare_survival` (log-rank test), `explain_survival`
  (Cox model, with a proportional-hazards check) and `chart.survival_curves` (with confidence
  bands, censoring ticks and a number-at-risk table).

### Planning a study

- `sdata.how_many_needed` and `sdata.chance_to_find`: power analysis for the averages, counts and
  correlation tests.

### More tests and summaries

- `compare_spreads` (Levene's test centred on medians, Bartlett or Fligner-Killeen),
  `compare_each_pair(..., test="tukey")` (Tukey's honest significant difference),
  `compare_counts_to` (chi-square goodness of fit) and `compare_percent_to` (exact binomial test).
- `percentile_of`, `most_common_of`, and new `summary_of(show=...)` measures: geometric mean,
  skewness, kurtosis, coefficient of variation, median absolute deviation, trimmed average, most
  common value and any percentile.
- Formulas know `exp`, `sin`, `cos`, `tan`, `pi` and special functions: `erf`, `erfc`, `gamma`,
  `lgamma`, `digamma`, `beta` and `besselJ`.

### Charts

- `colour_scale` chooses the colours for numbers (heatmaps, contour maps, dots coloured by a
  number): named colour-blind-safe scales, your own colours, `middle=`, `log=True`, `steps=` and
  `between=`.
- `transparency=` on every chart type (also survival curves) and `thickness=` on lines.
- `tick_format_x` / `tick_format_y` (percent, thousands, decimals, scientific, dates), and
  `ticks_x(every=...)` / `ticks_y(how_many=...)` for tick spacing, also by year, month or week.
- Histogram options: `show="percent"` or `"density"`, `stacked=`, `steps=`, `cumulative=` and
  `bin_width=`.
- `line_with_slope` (for 1:1 lines), `qq_dots` (Q-Q plots), `cumulative_share` (ECDF) and
  `font_name`.

### Choosing parts by position

- Printed tables number their rows from 1 (`show`, `show_first_rows`, `show_last_rows`, `print`,
  notebooks), so the last row of a 54-row table is row 54; `show_rows_with_outliers` shows each row
  with its number in the whole table. Only the display changes: `as_pandas()` keeps pandas' own
  index.
- `table.keep_rows_by_position(10, to=20)` and `remove_rows_by_position`: rows by their place,
  counting from 1 with both ends included (rows 10 to 20 are 11 rows), `to="last"`, `step=` for every
  n-th row and a list such as `[1, 5, 9]` (as pandas `iloc` and Polars `slice`/`gather_every`).
  `keep_columns_by_position` and `remove_columns_by_position` do the same for columns and say which
  columns they were.
- `table.row(3)` gives one row as a dict, `table.cell_at(row=3, column="marks")` one cell, and
  `table.reverse_rows()` turns the table upside down.
- `keep_columns` and `remove_columns` take a run of names (`keep_columns("hours", to="height")`),
  `kind="number"` (or `"text"`, `"date"`, `"true/false"`), `starts_with=`, `ends_with=`, `contains=`
  and `pattern=`.
- `replace_values` takes the `keep_rows` conditions: `replace_values("marks", above=100,
  with_value=100)`.
- Numbers: `nums.keep_numbers_by_position(3, to=8)`, `remove_numbers_by_position`, and
  `nums.replace_numbers(-999, with_value=None)` or `replace_numbers(above=100, with_value=100)`.
- Grids: `keep_rows_by_position`, `keep_columns_by_position`, `remove_rows_by_position`,
  `remove_columns_by_position` (a block is two steps) and `grid.replace_numbers`. `grid.keep_numbers`
  explains why a grid cannot drop single numbers.
- `"last"` is a position everywhere, also in `nums.number_at`, `grid.row`, `grid.column` and
  `grid.number_at`.
- `table[...]`, `table.marks` and loops over a table explain the words to use instead;
  `iloc`, `iat`, `select_dtypes`, `clip`, `np.where` and other names point to the new words.
- Lesson 10 teaches choosing by position, with a task to do from memory.
- Not yet: setting one cell by position, positions of matching numbers (argwhere, argmax), clip as
  one word, random fractions and sampling with replacement, grid diagonals and flips, unions of
  column choices, nested columns.

### Also

- Example tables `school`, `meadow` and `trial`; lessons 16 to 20; the new words are also taught
  in lessons 3, 5, 6, 11, 12 and 14. Lessons 19 and 20 end with a task that asks you to recall
  the words without looking back.
- The optional extra `stats` (statsmodels), needed only for mixed models and multiple
  imputation.
- Panel letters on a page no longer cover the panel titles of a chart made with
  `one_chart_per`.

## 0.1.0 (2026-11-12)

The first release: charts, tables, numbers and statistical tests in one small vocabulary.

### Charts

- About 70 words over matplotlib and seaborn: `bars`, `lines`, `dots`, `histogram`, `boxes`,
  `violins`, `heatmap`, `pie`, `error_bars`, `fit_line`, `smooth_line`, `density` and more.
- Titles, labels, legends, ranges, log scales, lines and shaded bands, notes with arrows,
  second y-axis, zoom insets, polar and 3D charts, animation.
- Panels: `sdata.grid(a, b, letters=True)` and `one_chart_per(down=, across=, per_row=)`.
- Months and weekdays in calendar order; `label_legend`, `names_on_dots`, `band`.
- Saving for journals: `chart.size(width=17, height=8, unit="cm")` (inches, cm, mm or pixels)
  gives a file of exactly that size, and `chart.save("figure.tif", resolution=600)` sets the dots
  per inch.
- Charts say how many rows with empty cells they left out; `chart.empty_cells()` shows where
  the gaps are.

### Tables

- Tables on pandas or Polars (through narwhals), with `undo()` and `start_again()`.
- Reading CSV, Excel, whole folders and web addresses; decimal commas and codes such as "NA";
  Word, SPSS and Stata files with the `files` extra.
- Choosing rows, including written conditions: `keep_rows("marks > 50 and class == 'A'")`.
  An empty cell never meets a condition.
- Cleaning: `remove_rows_with_empty_cells`, `fill_empty_cells` (also
  `with_the="line between neighbours"`), `add_is_empty_column`, outliers by Tukey's rule.
- Dates (`make_dates(order=)`, parts of dates, days between dates, `after=`/`before=`), text
  (split, join), rolling averages, previous values and changes, shares per group.
- Summaries: `average_of`, `summary_of`, totals per week, month or year (`every=`), cross
  tables and correlation tables.

### Numbers

- `sdata.numbers`, `sdata.number_grid` and `sdata.random_numbers` over NumPy, with plain words
  and `undo()`.

### Statistical tests

- `compare_averages` (Welch's t-test or Welch's ANOVA; rank tests on request),
  `compare_each_pair` (Holm-adjusted), `compare_before_after`, `compare_average_to`,
  `compare_counts` (chi-square, or Fisher's exact test), `compare_shape_to`,
  `fit_straight_line`, `fit_formula`, adjusted p-values.
- Effect sizes: Hedges' g, omega-squared, Cramér's V. Results checked against R 4.6.2.

### Learning

- `sdata.lesson(1)` to `sdata.lesson(15)`, `sdata.words()`, `sdata.help("word")`,
  `sdata.example()` tables (marks, weather, fruit, growth, island, shop), worked solutions.
- A cheat sheet (`docs/cheatsheet.md`) and a manual (`docs/manual/simplydata_manual.pdf`),
  both made from the code.
