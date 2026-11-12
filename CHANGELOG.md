# Changes

Each release lists what was added, changed or fixed. When a word is renamed, the old word keeps
working for one release and prints a one-line note with its new name.

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
