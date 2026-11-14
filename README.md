# simplydata

[![PyPI](https://img.shields.io/pypi/v/simplydata)](https://pypi.org/project/simplydata/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24210045.svg)](https://doi.org/10.5281/zenodo.24210045)

**Simple words for data science in Python.** One small vocabulary you can learn in a day,
built on matplotlib, seaborn, pandas, Polars, NumPy and SciPy (with narwhals linking pandas and
Polars), so you don't have to learn them first.

> Status: early development (charts, tables, numbers, statistical tests, fitting and distributions ready). The word list may still change: an old word keeps working for one release and prints a one-line note with its new name.

Install: `pip install simplydata`

```python
import simplydata as sdata

table = sdata.table(sdata.example("marks"))   # a made-up table; your own: sdata.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.average_of("marks", for_each="class")   # says the answer in a sentence
table.save("good_marks.xlsx")

chart = table.chart()                   # or sdata.chart(table)
chart.bars(x="class", y="marks", error="standard error")
chart.title_top("Good marks in each class")
chart.label_y("Marks")
chart.save("marks.png")
```

Tables, charts and numbers are separate objects, as in pandas, Polars, NumPy and matplotlib:
one cleaned table can feed many charts, and charts go side by side with
`sdata.grid(a, b, letters=True)` (panels A, B, C ... as in papers).

## Install

```
pip install simplydata            # needs Python 3.10 or newer
pip install "simplydata[fast]"    # adds Polars, the fast engine for big data
pip install "simplydata[files]"   # adds reading Word, SPSS and Stata files
pip install "simplydata[stats]"   # adds statsmodels, for mixed models and multiple imputation
```

## Learn it in a day

simplydata teaches itself. Open Python and type:

```python
import simplydata as sdata
sdata.lesson(1)
```

There are twenty short lessons. Lessons 1 to 8 teach charts, from your first chart to
animations. Lessons 9 to 12 teach tables: looking at your data, choosing rows and columns,
cleaning it, and summing it up. Lesson 13 teaches numbers (lists and grids of numbers), and
lesson 14 the statistical tests: is a difference real, or could it be chance? Lesson 15 puts it
all together, from a question and a file to a figure with panels A and B and a saved table.
Lessons 16 to 20 go further: models (multiple, logistic, Poisson and mixed regression, multiple
imputation), patterns across many columns (PCA, clustering, Bray-Curtis distances, NMDS,
PERMANOVA), survival curves and Cox models, power analysis, fitting curves and formulas,
finding the smallest or biggest value and solving equations, distributions and chances, and
values between the points you measured. Each
lesson has an example you can run, says what every line does, explains the idea behind it
(average or median, standard deviation or standard error, p-values) and ends with a small task
to try yourself.

- `sdata.lesson()` lists the lessons
- `sdata.words()` lists every word, with an example (also in [docs/cheatsheet.md](https://github.com/bioforkgithub/simplydata/blob/main/docs/cheatsheet.md))
- `sdata.help("bars")` explains one word: what it does, its options and an example
  (table and number words too: `sdata.help("keep_rows")`, `sdata.help("average")`)
- `sdata.example()` lists the made-up tables to practise with (marks, weather, fruit, growth, shop, school, meadow, trial, island; shop has dates and empty cells)
- `sdata.lesson(3, solutions=True)` adds a worked solution to each "Try it yourself" task

## Or learn it in 30 days

Rather learn an hour a day? simplydata has a 30-day course built in:

```python
import simplydata as sdata
sdata.start_30_days()                    # you already write some Python
sdata.start_30_days(track="beginner")    # you have never programmed: starts with just enough Python
sdata.today()                            # every day: the day to do next
```

Each day starts with a 5-minute warm-up of questions on earlier days (answered from memory, which
fixes words better than reading them again), then the lesson to work through, and ends with a task
to do without looking back. Review days mix the whole course, the last week is a project of your
own, and day 29 is a final test from memory. Missed a day? `sdata.today()` carries on where you
stopped. `sdata.answers()` gives the answers, `sdata.day()` the whole calendar, and `sdata.next_day()`
goes on now. Your progress is kept only on your computer, in `~/.simplydata/progress.json`.

**[30 days of simplydata (PDF)](https://github.com/bioforkgithub/simplydata/blob/main/docs/manual/simplydata_30_days.pdf)**
is the same course on paper: a calendar to tick for each track, one page per day with the page of
the manual for each lesson and word, and the answers at the back.

## Manual

**[The simplydata manual (PDF)](https://github.com/bioforkgithub/simplydata/blob/main/docs/manual/simplydata_manual.pdf)** has everything in one book:
all twenty lessons with the real charts and output they make, every word explained (its options,
an example you can run and the library that does the work underneath), the data science ideas
behind the numbers (average or median, standard deviation or standard error, confidence intervals,
p-values, effect sizes, and which statistical test to use when), and a worked solution for every
task.

**[The simplydata dictionary (PDF)](https://github.com/bioforkgithub/simplydata/blob/main/docs/manual/simplydataDictionary.pdf)** lists every word A to Z, with its meaning, an example and the page of the manual that explains it, and turns names from other tools (`groupby`, `iloc`, `hist`, `curve_fit` ...) into simplydata words.

The manual is made from the code by `python docs/make_manual.py` (it needs a LaTeX system with
`pdflatex`), so it never goes out of date: a test fails if
[docs/manual/simplydata_manual.tex](https://github.com/bioforkgithub/simplydata/blob/main/docs/manual/simplydata_manual.tex) no longer matches the code. The same command builds the dictionary and the 30-day guide,
with the page numbers of the manual it has just compiled.

## Tables

Plain words for looking at data, cleaning it and summing it up, on pandas or Polars:

```python
table = sdata.table(sdata.example("marks"))      # or sdata.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.sort_by("marks", biggest_first=True)
table.average_of("marks", for_each="class")  # says the answer in a sentence
table.compare_averages("marks", for_each="class")   # real difference, or chance?
table.undo()                               # take the last step back
table.save("good_marks.xlsx")
chart = table.chart()                      # draw it
```

## Numbers

Plain words over NumPy for lists and grids of numbers:

```python
nums = sdata.numbers([12, 15, 9, 20, 14])
bigger = nums * 2                          # symbols make new numbers
nums.keep_numbers(above=10)                # words change the numbers you have ...
nums.undo()                                # ... and can be taken back
nums.average()
grid = sdata.number_grid([[3, 5, 2], [8, 6, 4]])
grid.total(for_each="row")
```

## Choosing parts

Rows, columns and numbers by their place, counting from 1 the way the printed row numbers do,
with both ends included:

```python
table = sdata.table(sdata.example("marks"))
table.keep_rows_by_position(10, to=20)       # rows 10 to 20 (pandas df.iloc[9:20])
table.keep_rows_by_position(1, step=2)       # every 2nd row
table.keep_columns("hours", to="height")     # a run of columns by name
table.keep_columns(kind="number")            # every number column
table.cell_at(row=3, column="marks")         # one cell
table.replace_values("marks", above=90, with_value=90)
grid = sdata.numbers(1, to=24).number_grid(columns=6)
grid.keep_rows_by_position(2, to=3).keep_columns_by_position(1, to=3)   # a block
```

## More statistics

Models, patterns across many columns, survival and planning a study, each checked against R:

```python
school = sdata.table(sdata.example("school"))
school.explain("marks", using=["hours", "sleep", "breakfast"])          # multiple regression
school.explain("passed", using=["hours"], outcome="yes or no")          # logistic regression
school.explain("marks", using=["hours"], grouped_by="school")           # mixed model

meadow = sdata.table(sdata.example("meadow"))
meadow.add_principal_components(how_many=2)                             # PCA
meadow.add_nmds(measure="bray-curtis")                                  # NMDS map
meadow.compare_profiles(for_each="grazed")                              # PERMANOVA

trial = sdata.table(sdata.example("trial"))
trial.compare_survival(time="days", event="died", for_each="treatment") # log-rank test
sdata.how_many_needed(effect_size=0.5)                                  # power analysis
```

## Fitting, finding and distributions

Fit a line, a curve or your own formula; find the best value; solve an equation; ask a
distribution for chances; estimate values between the points you measured:

```python
growth = sdata.table(sdata.example("growth"))
growth.fit_formula("population = a * exp(b * (year - 2000))", for_each="country")  # as R nls()
growth.values_at("population", at={"year": [2001, 2011]}, for_each="country")    # as R approx()

sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])        # -3, at x = 2 (as R optimize())
sdata.solve("x**3 - 2*x = 5", between=[0, 3])                # x = 2.09455 (as R uniroot())

dist = sdata.distribution("normal", average=50, standard_deviation=10)
dist.chance(above=80)                                         # 0.135 %, and the R call: pnorm(...)
school = sdata.table(sdata.example("school"))
school.compare_distributions("days_absent", distributions=("poisson", "negative binomial"))
```

Coming from another tool? Its names point to the simplydata word and never run:
`sdata.minimize` says "In simplydata this is sdata.find_smallest(...)", `table.groupby` points to
`for_each=`, `chart.hist` to `chart.histogram`.

## How simplydata works

simplydata does not implement statistical methods itself. Every calculation is done by
established, peer-reviewed packages: Python packages (SciPy, statsmodels, pandas, NumPy and
others), and R packages through rpy2 where Python has none. simplydata translates simple words
into those calls, with the right settings, and says in the result which method was used. What we
test is that translation: each answer is checked against the package underneath and against R.

## Ideas behind it

- **Talk to the program.** Each step reads like a short sentence: `chart.title_top(...)`,
  `chart.label_x(...)`, `chart.line_at_y(50)`.
- **What + where.** A few place words repeat everywhere: `top`, `bottom`, `x`, `y`, `at`, `between`.
- **Guess a word from its sibling.** `keep_rows` and `keep_first_rows`, `show_first_rows`;
  `average_of` and `show="average"`; `label_x` and `label_colour_bar`.
- **The word names the idea.** `error="standard error"`, `fit_line(confidence_band=True)`,
  `compare_averages` (and the result says "Welch's t-test" for when you write it up).
- **Order doesn't matter.** Write the steps in any order; simplydata draws them correctly.
- **Friendly errors.** Misspell a column and simplydata tells you what you probably meant.
- **Your old words still help.** Names from pandas, NumPy, SciPy, R and matplotlib (`groupby`,
  `curve_fit`, `pnorm`, `hist`) point to the simplydata word, so the first guess is never a dead end.

The full design is in [PLAN.md](https://github.com/bioforkgithub/simplydata/blob/main/PLAN.md).

## Changes and contributing

What changed in each release is in [CHANGELOG.md](https://github.com/bioforkgithub/simplydata/blob/main/CHANGELOG.md). Contributions are welcome:
[CONTRIBUTING.md](https://github.com/bioforkgithub/simplydata/blob/main/CONTRIBUTING.md) explains how to suggest a word, report a problem or add code.

## How to cite

If simplydata helps your work, please cite the archived software:

> Victor, M. P. simplydata: simple words for data science in Python. Zenodo. https://doi.org/10.5281/zenodo.24210045

This DOI always points to the latest version; each release also has its own DOI on Zenodo. GitHub's "Cite this repository" button gives the same citation in other formats.

## Licence

MIT.
