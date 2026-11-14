"""Build the simplydata manual (docs/manual/simplydata_manual.tex and .pdf) from the code.

Everything in the manual comes from simplydata itself: the lessons, their solutions and the
word list from simplydata/learn.py (LESSONS, HELP_EXAMPLES, GROUPS), each word's options and
text from its docstring and signature, and the practice tables from simplydata.data. Only
three things are written by hand: the short "Getting started" text below, UNDERNEATH (which
library each word uses) and docs/manual/ideas.tex (the data science ideas).

    python docs/make_manual.py            # write the .tex, run every example, compile the PDF
    python docs/make_manual.py --tex      # only write the .tex (fast; no LaTeX needed)
    python docs/make_manual.py --no-run   # write the .tex and compile, without running the examples again

Compiling the manual also builds the dictionary (docs/manual/simplydataDictionary.pdf, made by
docs/make_dictionary.py) and the guide "30 days of simplydata" (docs/manual/simplydata_30_days.pdf,
made by docs/make_30_days.py), so their page numbers always match the manual. The LaTeX build files go
to a scratch folder; set SIMPLYDATA_LATEX_DIR to choose another one.

Building the PDF needs a LaTeX system with pdflatex (MiKTeX or TeX Live). It runs every
lesson example and every solution, saves their charts in docs/manual/figures/ and what they
print in docs/manual/output/, then compiles the PDF three times (for the page numbers).
The .tex has no dates in it, so it only changes when the code changes; a test
(tests/test_learn.py) fails if docs/manual/simplydata_manual.tex is out of date.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import simplydata
from simplydata import learn
from simplydata.data import EXAMPLES

HERE = Path(__file__).resolve().parent
MANUAL = HERE / "manual"
TEX = MANUAL / "simplydata_manual.tex"
PDF = MANUAL / "simplydata_manual.pdf"
FIGURES = MANUAL / "figures"
OUTPUT = MANUAL / "output"

MAX_OUTPUT_LINES = 40  # longer printed output is cut, with a line that says how much was left out
IMAGE_TYPES = ("png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff", "gif")

# ---------------------------------------------------------------- which library is underneath
# One line per group: what most of its words use. A word whose library differs, or is worth
# naming more exactly, is in UNDERNEATH. make() stops with a message if a word has neither.
GROUP_UNDERNEATH = {
    "Lines and shading": "matplotlib",
    "Words on the chart": "matplotlib",
    "Axes": "matplotlib",
    "Colours and look": "matplotlib",
    "Order of groups": "pandas (a stable sort of the rows, so every chart type, colour and panel follows the order)",
    "Tables: start and look": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Choose columns": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Choose rows": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Empty cells": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Distributions": "SciPy (scipy.stats frozen distributions: cdf, sf, ppf, isf, pdf, pmf, stats, rvs)",
    "Fix columns": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Sort and new columns": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Dates and text": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Row after row": "narwhals (shift, rolling_mean and over with order_by=, so each group keeps its own rows)",
    "Summaries": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Combine and reshape": "narwhals, which runs the step in pandas or Polars (whichever holds the table)",
    "Undo and finish": "simplydata itself: it keeps every version of the table",
    "Numbers: make and look": "NumPy",
    "Numbers: maths in words": "NumPy",
    "Numbers: choose, sort and fill": "NumPy",
    "Numbers: summaries": "NumPy (functions that skip empty values, such as nanmean)",
    "Grids of numbers": "NumPy (a two-dimensional array)",
    "Numbers: undo and finish": "simplydata itself: it keeps every version of the numbers",
    "Help": "plain Python: simplydata reads its own word list and docstrings",
    "30 days of simplydata": "plain Python: the days are in simplydata/learn_days.py, and the progress is a small "
                             "JSON file on your computer (json, pathlib)",
}
UNDERNEATH = {
    # start, show and save
    "sdata.chart": "pandas reads the file (Polars for big files); the chart is drawn with matplotlib and seaborn",
    "sdata.example": "pandas (read_csv of a small file inside simplydata)",
    "show": "matplotlib (pyplot.show)",
    "save": "matplotlib (Figure.savefig); Pillow writes .gif animations",
    # chart types
    "bars": "seaborn (barplot; countplot for show=\"count\")",
    "lines": "seaborn (lineplot); SciPy (interpolate) for joined_by= curves",
    "dots": "seaborn (scatterplot)",
    "area": "matplotlib (fill_between, or stackplot with stacked=True)",
    "histogram": "seaborn (histplot)",
    "boxes": "seaborn (boxplot)",
    "violins": "seaborn (violinplot)",
    "heatmap": "seaborn (heatmap); SciPy (scipy.cluster.hierarchy) for cluster=True",
    "pie": "matplotlib (Axes.pie)",
    "polar": "matplotlib (polar axes)",
    "contour": "matplotlib (contourf); SciPy (interpolate.griddata) for scattered points",
    "surface": "matplotlib (mplot3d plot_surface); SciPy (interpolate.griddata) for scattered points",
    "dots_3d": "matplotlib (mplot3d scatter)",
    "band": "matplotlib (fill_between)",
    "empty_cells": "pandas finds the empty cells (isna); matplotlib (imshow) draws the grid",
    "names_on_dots": "matplotlib (annotate), with simplydata choosing a free place for each name",
    # statistics
    "show=": "seaborn (the estimator of barplot and lineplot)",
    "error=": "seaborn (errorbar=\"sd\" or \"se\"); the confidence interval from the t distribution (SciPy)",
    "error_bars": "pandas (group averages) and matplotlib (errorbar); SciPy for the t-based confidence interval",
    "fit_line": "SciPy (stats.linregress), drawn with matplotlib",
    "fit_curve": "NumPy (polyfit), drawn with matplotlib",
    "smooth_line": "NumPy (LOWESS, written out in simplydata as in R's lowess()) or SciPy (make_smoothing_spline, for "
                   "method=\"spline\"), drawn with matplotlib",
    "density": "seaborn (kdeplot)",
    "qq_dots": "SciPy (stats.probplot, against a fitted scipy.stats distribution for distribution=) for the points "
               "and the line; matplotlib draws them",
    "cumulative_share": "seaborn (ecdfplot)",
    "count_bars": "seaborn (countplot)",
    "all_dots": "seaborn (swarmplot; stripplot for groups of more than 100 dots)",
    "side_histograms": "matplotlib (hist), with NumPy choosing the bins",
    "sdata.pairs": "seaborn (scatterplot, histplot and kdeplot)",
    # several charts
    "one_chart_per": "matplotlib (a grid of subplots)",
    "sdata.side_by_side": "matplotlib (subplots)",
    "sdata.one_above_other": "matplotlib (subplots)",
    "sdata.grid": "matplotlib (subplots)",
    "zoom_box": "matplotlib (inset_axes and indicate_inset_zoom)",
    "animate": "matplotlib (animation) and Pillow (the .gif file)",
    # tables
    "sdata.table": "pandas or Polars read the file; narwhals lets every table word run on both",
    "table.show": "pandas or Polars (printing the table)",
    "table.column": "NumPy (a column of numbers becomes sdata.numbers)",
    "table.fill_empty_cells": "narwhals; NumPy (interp) for the line between neighbours; SciPy (CubicSpline, "
                              "PchipInterpolator) for the curves",
    "table.make_dates": "pandas (to_datetime) or Polars (str.strptime), trying the date formats in a fixed order",
    "table.split_column": "pandas (str.split) or Polars (str.splitn), whichever holds the table",
    "table.add_rolling_column": "narwhals (rolling_mean, rolling_sum, rolling_std with over(order_by=)); "
                                "pandas or Polars rolling for median, smallest and biggest",
    "table.summary_of": "narwhals (group summaries); SciPy (the t distribution) for the confidence interval",
    "table.percent_of": "narwhals; SciPy (stats.binomtest, the Clopper-Pearson exact interval)",
    "table.compare_spreads": "SciPy (stats.levene with center=\"median\", stats.bartlett, stats.fligner)",
    "table.compare_counts_to": "SciPy (stats.chisquare)",
    "table.compare_percent_to": "SciPy (stats.binomtest, the Clopper-Pearson exact interval)",
    "table.percentile_of": "narwhals (quantile with linear interpolation, as R quantile type 7)",
    "table.most_common_of": "narwhals (value counts per group)",
    "table.add_adjusted_p_value_column": "NumPy (Holm, Bonferroni, Benjamini-Hochberg and Benjamini-Yekutieli, "
                                         "written out as in R p.adjust())",
    "table.add_rows_from": "pandas (concat) or Polars (concat), whichever holds the table",
    "table.correlation_between": "SciPy (stats.pearsonr or stats.spearmanr)",
    "table.compare_averages": "SciPy (stats.ttest_ind with equal_var=False; Welch's ANOVA written out with "
                              "the F distribution; mannwhitneyu, kruskal)",
    "table.compare_each_pair": "SciPy (stats.ttest_ind or mannwhitneyu), with Holm's correction",
    "table.compare_before_after": "SciPy (stats.ttest_rel or stats.wilcoxon)",
    "table.compare_average_to": "SciPy (stats.ttest_1samp or stats.wilcoxon)",
    "table.compare_counts": "SciPy (stats.chi2_contingency, or fisher_exact with test=\"fisher\")",
    "table.compare_shape_to": "SciPy (stats.shapiro for normal; for other shapes the method of "
                              "stats.goodness_of_fit, Anderson-Darling with a Monte Carlo p-value, with the "
                              "simulated samples refitted all at once in NumPy)",
    # fitting
    "table.fit_line": "SciPy (stats.linregress)",
    "table.fit_curve": "NumPy (least squares by QR on the powers of x)",
    "table.fit_formula": "SciPy (optimize.curve_fit; optimize.differential_evolution for search=\"whole range\")",
    "fit_formula": "SciPy (optimize.curve_fit, as table.fit_formula), drawn with matplotlib",
    "fit_distribution": "SciPy (scipy.stats, maximum likelihood as table.fit_distribution), drawn with matplotlib",
    "curve": "NumPy (the formula on a grid of x values) or scipy.stats (a distribution), drawn with matplotlib",
    # finding and solving
    "sdata.find_smallest": "SciPy (optimize.minimize_scalar, minimize, linprog and milp with HiGHS, "
                           "differential_evolution and other global searches)",
    "sdata.find_biggest": "SciPy (the same as find_smallest, on minus the formula)",
    "sdata.solve": "SciPy (optimize.brentq, root_scalar, root) and NumPy (polynomial roots, linalg.solve)",
    # values between points
    "table.values_at": "NumPy (interp) and SciPy (interpolate: CubicSpline, PchipInterpolator, Akima1DInterpolator, "
                       "make_smoothing_spline, griddata, RegularGridInterpolator, RBFInterpolator)",
    "table.map_of": "SciPy (interpolate: RegularGridInterpolator, griddata, RBFInterpolator)",
    # distributions
    "sdata.distribution": "SciPy (scipy.stats frozen distributions)",
    "table.fit_distribution": "SciPy (scipy.stats and optimize, maximum likelihood; closed forms where they exist)",
    "table.compare_distributions": "SciPy (scipy.stats, maximum likelihood; Anderson-Darling, Kolmogorov-Smirnov and "
                                   "Cramer-von Mises statistics)",
    "nums.fit_distribution": "SciPy (scipy.stats and optimize, as table.fit_distribution)",
    # models
    "table.explain": "NumPy (least squares, and iteratively reweighted least squares as R glm); "
                     "statsmodels for grouped_by= (MixedLM) and multiple imputation (MICEData)",
    "table.add_predicted_column": "NumPy (the same model as table.explain); statsmodels for grouped_by=",
    # patterns across columns
    "table.add_principal_components": "NumPy (linalg.svd)",
    "table.principal_components_of": "NumPy (linalg.svd)",
    "table.add_clusters": "NumPy (k-means: Lloyd and Hartigan steps) or SciPy (cluster.hierarchy, for "
                          "method=\"hierarchical\")",
    "table.distances_between_rows": "SciPy (spatial.distance.pdist) and NumPy",
    "table.add_nmds": "NumPy (SMACOF with monotone regression; SciPy optimize.isotonic_regression when "
                      "available) and SciPy (spatial.distance.pdist)",
    "table.compare_profiles": "NumPy (PERMANOVA and the spread check, with permutations) and SciPy "
                              "(spatial.distance.pdist)",
    # survival
    "table.survival_over_time": "NumPy (Kaplan-Meier) and SciPy (stats, for the intervals)",
    "table.compare_survival": "NumPy (log-rank sums) and SciPy (stats.chi2)",
    "table.explain_survival": "NumPy (Cox model: Newton-Raphson with Efron ties) and SciPy (stats)",
    "survival_curves": "NumPy (Kaplan-Meier), drawn with matplotlib",
    # planning a study
    "sdata.how_many_needed": "SciPy (stats.nct, ncf, ncx2 and norm, and optimize.brentq)",
    "sdata.chance_to_find": "SciPy (stats.nct, ncf, ncx2 and norm)",
    "table.save": "pandas (to_excel, to_csv ...) or Polars, whichever holds the table",
    "table.chart": "matplotlib and seaborn (the chart is drawn from a pandas copy of the table)",
    "table.as_pandas": "pandas",
    "table.as_polars": "Polars",
    # numbers
    "sdata.random_numbers": "NumPy (random.default_rng)",
    "nums.percentile": "NumPy (nanpercentile, the usual \"linear\" method, as pandas and R)",
    "nums.table": "pandas",
    "nums.chart": "matplotlib and seaborn, from a pandas table with one column called value",
    "nums.save": "pandas (to_csv, to_excel ...)",
    "nums.keep_random_numbers": "NumPy (random.default_rng)",
    "nums.fill_empty_numbers": "NumPy (interp for the line between neighbours)",
}

# ---------------------------------------------------------------- hand-written text
GETTING_STARTED = r"""
\unchapter{Welcome}{start:welcome}

\textbf{simplydata} gives you one small vocabulary of plain English words for working with data:
reading a table, cleaning it, summing it up, testing a difference and drawing a chart. Underneath,
the work is done by the libraries that scientists use every day: pandas and Polars for tables,
NumPy for numbers, SciPy for statistics, and matplotlib and seaborn for charts. You learn the
words; simplydata picks the right call in the right library, with the right settings.

\begin{lstlisting}[style=py]
import simplydata as sdata

table = sdata.table(sdata.example("marks"))   # or your own: sdata.table("marks.xlsx")
table.keep_rows("marks", above=50)
table.average_of("marks", for_each="class")   # says the answer in a sentence
table.save("good_marks.xlsx")

chart = table.chart()                         # or sdata.chart(table)
chart.bars(x="class", y="marks", error="standard error")
chart.title_top("Good marks in each class")
chart.save("marks.png")
\end{lstlisting}

The program reads like a list of instructions, one step on each line. A table, a chart and a list
of numbers are separate objects, as in pandas, NumPy and matplotlib: one cleaned table can feed
many charts, and several charts can share one page.

\section*{Who this manual is for}
Everyone who wants to learn data science with Python: school pupils, students, teachers and
researchers. You do not need to know Python first. Every word is simple enough for a child to read,
and every result names the technical term (``Welch's t-test'', ``linear regression'') so that you
can write it up in a report or a paper.

\section*{How this manual is built}
\begin{description}[style=nextline,leftmargin=1.2em]
  \item[{\hyperref[part:lessons]{Part II: Lessons}}] @LESSON_COUNT@ short lessons that teach the whole
    of simplydata in a day. Each lesson shows code you can run, says what every line does, shows
    the real chart and output it makes, and ends with tasks to try yourself.
  \item[{\hyperref[part:dictionary]{Part III: Dictionary of words}}] Every word, with what it does,
    its options, an example you can run and the library it uses underneath.
  \item[{\hyperref[part:ideas]{Part IV: Data science ideas}}] The ideas behind the numbers in plain
    words: average and median, standard deviation and standard error, confidence intervals,
    p-values, effect sizes, and a guide to which statistical test to use when.
  \item[{\hyperref[part:solutions]{Part V: Solutions}}] A worked solution for every
    ``Try it yourself'' task.
\end{description}
Everything in this manual is made from simplydata's own code by \code{docs/make\_manual.py}: the
examples were run to make the charts and the printed output you see, so they are exactly what you
will get.

\unchapter{Install and first steps}{start:install}

\section*{Install}
\begin{lstlisting}[style=plain]
pip install simplydata            # coming soon
pip install simplydata[fast]      # adds Polars, the fast engine for big data
\end{lstlisting}

\section*{Where to type the code}
Any place that runs Python works. In \textbf{Jupyter}, \textbf{VS Code} or \textbf{Spyder}, a chart
appears by itself when you write \code{chart} on the last line of a cell. In a script,
\code{chart.show()} opens it in a window, and \code{chart.save("name.png")} always saves it as a file.

\section*{simplydata teaches itself}
\begin{lstlisting}[style=py]
import simplydata as sdata
sdata.lesson()                      # lists the lessons
sdata.lesson(1)                     # the first lesson
sdata.lesson(1, solutions=True)     # the first lesson, with worked solutions to its tasks
sdata.words()                       # every word, with a short example
sdata.help("bars")                  # one word: what it does, its options and an example
sdata.example()                     # the made-up tables you can practise with
sdata.start_30_days()               # or learn it in 30 days of about an hour,
sdata.today()                       # ... with sdata.today() each day
\end{lstlisting}
Rather learn an hour a day? \code{sdata.start\_30\_days()} spreads the lessons over 30 days, each with
a warm-up of questions on earlier days and a task, and a project of your own at the end
(\code{track="beginner"} starts with just enough Python). The printable guide,
\href{simplydata_30_days.pdf}{\code{simplydata\_30\_days.pdf}}, has the same days on paper.

\section*{Your own data}
\code{sdata.table("my\_file.xlsx")} and \code{sdata.chart("my\_file.xlsx")} read Excel, CSV, TSV,
Parquet and JSON files. They also take a pandas or Polars table you already have, or a small table
typed by hand: \code{sdata.table(\{"name": ["Asha", "Ben"], "height": [144, 152]\})}. Your file is
never changed: only \code{table.save(...)} writes a file.

\section*{When something goes wrong}
simplydata explains its errors in plain words. Misspell a column and it says which columns there
are and which one you probably meant. Ask for a word that does not exist and \code{sdata.help}
suggests the nearest one. A missing word can be requested on GitHub, at
\url{https://github.com/bioforkgithub/simplydata/issues}.

\unchapter{How the words are built}{start:grammar}

A few rules make the words easy to guess. If you know one word of a pattern, you can usually
guess its siblings.

\begin{center}
\small
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.33\textwidth}>{\raggedright\arraybackslash}X@{}}
\toprule
\textbf{Pattern} & \textbf{Examples}\\
\midrule
\textbf{verb + thing} changes a table or numbers
  & \code{keep\_rows}, \code{remove\_columns}, \code{fill\_empty\_cells}, \code{add\_column}, \code{keep\_numbers}\\
\textbf{chart types are nouns} on a chart
  & \code{chart.bars(...)}, \code{chart.histogram(...)}, \code{chart.dots(...)}\\
\textbf{thing + where} places something on a chart
  & \code{title\_top}, \code{label\_y}, \code{line\_at\_y}, \code{legend\_bottom}, \code{shade\_between\_x}\\
\textbf{measure + \_of} sums up one column; \textbf{relation + \_between} links two;
\textbf{compare\_ + measure} tests
  & \code{average\_of}, \code{median\_of}, \code{correlation\_between}, \code{compare\_averages}\\
\textbf{show\_ / hide\_ + thing} looks without changing
  & \code{show\_first\_rows}, \code{show\_grid}, \code{hide\_legend}\\
\textbf{as\_ + library} hands the data on; \textbf{save} writes a file
  & \code{as\_pandas}, \code{as\_polars}, \code{as\_numpy}; \code{table.save("clean.xlsx")}\\
\bottomrule
\end{tabularx}
\end{center}

\begin{itemize}
  \item \textbf{One option name, one meaning, everywhere:} \code{for\_each=} (groups),
    \code{colour=} (one colour, or a colour per group), \code{between=[low, high]}, \code{from\_column=},
    \code{with\_value=}, \code{biggest\_first=}, \code{per\_row=}, \code{label=}, \code{fast=}.
  \item \textbf{Measure names are the same wherever they appear:} \code{"average"}, \code{"median"},
    \code{"total"}, \code{"count"}, \code{"smallest"}, \code{"biggest"}, \code{"standard deviation"},
    \code{"standard error"}, \code{"confidence interval"}.
  \item \textbf{Axes are x and y.} Anything that belongs to one axis ends in \code{\_x} or \code{\_y}:
    \code{label\_y}, \code{range\_x}, \code{log\_scale\_y}.
  \item \textbf{The word names the idea}, and the printed answer names the technical term, so you can
    follow (and write) the Methods section of a paper.
  \item \textbf{Order does not matter} on a chart: after the chart type, write the other steps in any
    order. Both \code{colour} and \code{color}, \code{grey} and \code{gray} work.
  \item \textbf{Every change can be taken back}: tables and numbers remember each step, so
    \code{undo()}, \code{steps()} and \code{start\_again()} always work.
\end{itemize}
"""

# Family of chapters in the dictionary: chosen from the kind of the words in each group.
FAMILY_TITLES = {
    "chart": ("Chart words", "Steps on a chart: \\code{chart = sdata.chart(table)}, then "
                             "\\code{chart.bars(...)}. Words that start with \\code{sdata.} are called on "
                             "simplydata itself, and words that end in \\code{=} are options inside a chart step."),
    "table": ("Table words", "Steps on a table: \\code{table = sdata.table(...)}, then "
                             "\\code{table.keep\\_rows(...)}. Table words work the same on pandas and Polars tables. "
                             "Words that start with \\code{sdata.} (finding and solving, planning a study) are "
                             "called on simplydata itself, and words that start with \\code{dist.} are steps on "
                             "a distribution: \\code{dist = sdata.distribution(\"normal\")}."),
    "numbers": ("Number words", "Steps on numbers, \\code{nums = sdata.numbers([12, 15, 9, 20, 14])}, "
                                "and on grids of numbers, \\code{grid = sdata.number\\_grid([[3, 5, 2], [8, 6, 4]])}."),
    "help": ("Help words", "Words that teach simplydata."),
}


# ---------------------------------------------------------------- LaTeX text
_UNICODE = {
    "°": r"\textdegree{}", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "⁴": r"\textsuperscript{4}", "·": r"\textperiodcentered{}",
    "×": r"\texttimes{}", "−": r"\textminus{}", "±": r"\textpm{}", "–": "--", "—": "---", "…": r"\dots{}",
    "≈": r"$\approx$", "µ": r"\textmu{}", "é": r"\'e", "’": "'", "‘": "`", "“": "``", "”": "''",
}
_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
    "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", '"': r"\textquotedbl{}",
}


def esc(text: str) -> str:
    """Escape plain text for LaTeX: special characters and the few non-ASCII ones we use."""
    out = []
    for ch in str(text):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        elif ord(ch) > 126:
            raise ValueError(f"Add {ch!r} (U+{ord(ch):04X}) to _UNICODE in docs/make_manual.py")
        else:
            out.append(ch)
    text = "".join(out)
    return text.replace("--", "-{}-").replace("``", "`{}`").replace("''", "'{}'")


def code(text: str) -> str:
    """Inline code: typewriter font, escaped. A long piece may break after a dot, comma or
    underscore, so it never sticks out into the margin."""
    text = esc(text)
    for mark in (".", ",", r"\_"):
        text = text.replace(mark, mark + r"\allowbreak{}")
    return r"\code{" + text + "}"


# Code inside prose: a call such as chart.bars(...), an option such as show="median", a
# "quoted value", a dotted or underscored name (table.save, keep_rows) or a file ending (.xlsx).
_CALL = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\(")
_OPTION = re.compile(r"(?<![\w.])[a-z_][a-z0-9_]*=(?!=)")
_QUOTED = re.compile(r'"[^"\n]*"')
_NAME = re.compile(r"(?<![\w.])(?:(?:sdata|chart|table|nums|grid|page)\.\w+(?:\(\))?|[A-Za-z]\w*_\w*(?:\(\))?)")
_ENDING = re.compile(r"(?<![\w])\.(?:png|pdf|svg|jpg|tif|gif|xlsx|csv|tsv|parquet|json)\b")


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """Index just after the bracket that closes the one at text[start], quotes respected."""
    depth, quote = 0, None
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _code_span(text: str, i: int) -> int | None:
    """If a piece of code starts at text[i], the index where it ends; else None."""
    m = _CALL.match(text, i)
    if m:
        end = _balanced(text, m.end() - 1, "(", ")")
        if end:
            return end
    m = _OPTION.match(text, i)
    if m:
        j = m.end()
        if j < len(text) and text[j] == '"':
            close = text.find('"', j + 1)
            return close + 1 if close > 0 else j
        if j < len(text) and text[j] == "[":
            return _balanced(text, j, "[", "]") or j
        value = re.match(r"[\w.\-]*", text[j:])
        return j + len(value.group(0).rstrip("."))
    for pattern in (_QUOTED, _NAME, _ENDING):
        m = pattern.match(text, i)
        if m:
            return m.end()
    return None


def rich(text: str) -> str:
    """Prose with the pieces of code in it set in typewriter font."""
    out, plain, i = [], [], 0
    while i < len(text):
        end = _code_span(text, i) if (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) else None
        if end:
            out.append(esc("".join(plain)))
            plain = []
            out.append(code(text[i:end]))
            i = end
        else:
            plain.append(text[i])
            i += 1
    out.append(esc("".join(plain)))
    return "".join(out)


def label(key: str) -> str:
    """A LaTeX label for a word: "table.keep_rows" -> "word:table.keep-rows"."""
    return "word:" + key.replace("_", "-").replace("=", "-eq")


def listing(source: str, style: str = "py") -> str:
    return f"\\begin{{lstlisting}}[style={style}]\n{source.rstrip()}\n\\end{{lstlisting}}"


# ---------------------------------------------------------------- figures and printed output
_SAVE = re.compile(r'\.save\("([^"]+)"')


def saved_images(source: str) -> list[tuple[str, str]]:
    """(stem, type) of each picture the code saves, once per name: a chart saved as .png and .pdf
    is one picture."""
    seen, found = set(), []
    for name in _SAVE.findall(source):
        stem, _, kind = name.rpartition(".")
        if kind.lower() in IMAGE_TYPES and stem not in seen:
            seen.add(stem)
            found.append((stem, kind.lower()))
    return found


def figure_files(prefix: str, source: str) -> list[str]:
    """The files (relative to docs/manual) that hold the pictures of one example."""
    return [f"figures/{prefix}_{stem}.{'png' if kind == 'gif' else 'pdf'}" for stem, kind in saved_images(source)]


def figure_block(prefix: str, source: str, caption: str, width: float, height: float,
                 numbered: bool = True) -> str:
    """The pictures an example saved, side by side, with a caption that names the files."""
    files = figure_files(prefix, source)
    if not files:
        return ""
    names = [f"{stem}.{kind}" for stem, kind in saved_images(source)]
    each = width if len(files) == 1 else min(width, 0.96 / len(files))
    pics = "\\hfill".join(
        f"\\includegraphics[width={each:.2f}\\textwidth,height={height:.2f}\\textheight,keepaspectratio]{{{f}}}"
        for f in files)
    what = " and ".join(code(n) for n in names)
    gif = any(n.endswith(".gif") for n in names)
    note = " (the last picture of the animation)" if gif else ""
    star = "" if numbered else "*"
    return ("\\begin{center}\\begin{minipage}{\\linewidth}\\centering\n" + pics
            + f"\n\\captionof{star}{{figure}}{{" + caption + " " + what + note
            + ".}\n\\end{minipage}\\end{center}")


def output_block(prefix: str) -> str:
    return f"\\printedoutput{{output/{prefix}.txt}}"


def _lesson_prefix(number: int, part: int) -> str:
    return f"lesson{number:02d}_example{part}"


def _solution_prefix(number: int, task: int) -> str:
    return f"solution{number:02d}_task{task}"


def jobs() -> list[tuple[str, str]]:
    """(prefix, code) for every lesson example and every solution: what the build runs."""
    found = []
    for number in range(1, len(learn.LESSONS) + 1):
        for part, source in enumerate(learn.lesson_code(number), start=1):
            found.append((_lesson_prefix(number, part), source))
        for task, (source, _) in enumerate(learn.lesson_solutions(number), start=1):
            found.append((_solution_prefix(number, task), source))
    return found


# ---------------------------------------------------------------- the parts of the book
HEADER = """\
% ---------------------------------------------------------------------------------------------
% The simplydata manual. This file is made by docs/make_manual.py from simplydata's own code.
% Do not edit it by hand: change simplydata/learn.py, a docstring, docs/manual/preamble.tex or
% docs/manual/ideas.tex, then run   python docs/make_manual.py
% ---------------------------------------------------------------------------------------------"""


def _preamble() -> str:
    return HEADER + "\n" + (MANUAL / "preamble.tex").read_text(encoding="utf-8").strip()


def _title_page() -> str:
    return r"""
\begin{titlepage}
\begin{tikzpicture}[remember picture,overlay]
  \fill[brand] (current page.north west) rectangle ([yshift=-9.5cm]current page.north east);
\end{tikzpicture}
\vspace*{1.2cm}
{\color{white}\sffamily
{\fontsize{46}{50}\selectfont\bfseries simplydata\par}
\vspace{0.5cm}
{\LARGE simple words for data science\par}
\vspace{0.4cm}
{\large The manual: lessons, every word, the ideas behind them, and solutions\par}}
\vspace{3.2cm}
\begin{lstlisting}[style=title]
import simplydata as sdata

table = sdata.table(sdata.example("marks"))
table.compare_averages("marks", for_each="class")
chart = table.chart()
chart.boxes(x="class", y="marks")
chart.save("marks.png")
\end{lstlisting}
\vfill
{\sffamily
\textbf{Version @VERSION@}\\[0.3em]
Open source (MIT licence): \url{https://github.com/bioforkgithub/simplydata}\\[0.3em]
Made from the code by \code{docs/make\_manual.py}; every example in it was run to make its chart.\par}
\end{titlepage}
""".replace("@VERSION@", esc(simplydata.__version__))


def _practice_tables() -> str:
    lines = [r"\unchapter{The practice tables}{start:tables}",
             "All the examples use small made-up tables that come with simplydata, so they run "
             "anywhere without a file. Every number in them is invented. Get one with "
             + code('sdata.example("marks")') + ", and list them all with " + code("sdata.example()") + "."]
    for name, (about, columns) in EXAMPLES.items():
        lines.append(f"\\section*{{{esc(name)}}}")
        lines.append(rich(about) + " " + code(f'sdata.example("{name}")'))
        lines.append("\\begin{center}\\small")
        lines.append(r"\begin{tabularx}{\textwidth}{@{}>{\ttfamily}l>{\raggedright\arraybackslash}X@{}}")
        lines.append(r"\toprule \textrm{\textbf{Column}} & \textbf{What it holds}\\ \midrule")
        for column, meaning in columns.items():
            lines.append(f"{esc(column)} & {rich(meaning)}\\\\")
        lines.append(r"\bottomrule\end{tabularx}\end{center}")
    return "\n".join(lines)


def _lesson(number: int, aliases_lookup) -> str:
    les = learn.LESSONS[number - 1]
    out = [f"\\chapter{{{esc(les['title'])}}}\\label{{lesson:{number}}}"]
    out.append(r"\begin{learnbox}")
    out.append(r"\begin{itemize}")
    out += [f"  \\item {rich(item)}" for item in les["learn"]]
    out.append(r"\end{itemize}")
    out.append(r"\end{learnbox}")
    parts = les["parts"]
    for i, (intro, steps) in enumerate(parts, start=1):
        heading = "Example" if len(parts) == 1 else f"Example {i}"
        out.append(f"\\section*{{{heading}}}")
        out.append(rich(intro))
        source = "\n".join(c for c, _ in steps)
        out.append(listing(source))
        out.append(r"\whateachline")
        out.append(r"\begin{lineexplain}")
        for line_no, (_, explain) in enumerate(steps, start=1):
            if explain:
                out.append(f"  \\item[{line_no}] {rich(explain)}")
        out.append(r"\end{lineexplain}")
        prefix = _lesson_prefix(number, i)
        out.append(output_block(prefix))
        fig = figure_block(prefix, source, f"Lesson {number}, {heading.lower()}:", 0.72, 0.36)
        if fig:
            out.append(fig)
    if les.get("notes"):
        out.append(r"\begin{goodtoknow}")
        out.append(r"\begin{itemize}")
        out += [f"  \\item {rich(note)}" for note in les["notes"]]
        out.append(r"\end{itemize}")
        out.append(r"\end{goodtoknow}")
    out.append(r"\begin{tryit}")
    out.append(r"\begin{enumerate}")
    out += [f"  \\item {rich(task)}" for task in les["task"]]
    out.append(r"\end{enumerate}")
    out.append(f"Worked solutions: page~\\pageref{{solutions:{number}}}, "
               f"or type {code(f'sdata.lesson({number}, solutions=True)')}.")
    out.append(r"\end{tryit}")
    words = []
    for word in les["words"]:
        key = learn._lookup(word)
        words.append(f"\\hyperref[{label(key)}]{{{code(learn._call_form(key))}}}")
    out.append(r"\par\noindent{\small\sffamily\textbf{Words in this lesson:} " + ", ".join(words) + ".\\par}")
    return "\n".join(out)


def _options_table(key: str) -> str:
    """The options in the brackets, as sdata.help shows them: "needed", "normally ..." or nothing
    (no fixed value: the text above says what it does)."""
    func = learn._function(key)
    if func is None:
        return ""
    params = [p for p in inspect.signature(func).parameters.values()
              if p.name != "self" and p.kind is not p.VAR_KEYWORD
              and not (p.kind is p.VAR_POSITIONAL and p.name == "extra")]
    if not params:
        return r"\wordfield{Options}{none: just write the empty brackets.}"
    rows = []
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            rows.append((p.name + "...", "as many as you like, separated by commas"))
        elif p.default is p.empty:
            rows.append((p.name, "needed"))
        elif p.default is None or type(p.default).__name__ == "_NotGiven":
            rows.append((p.name + "=", ""))
        else:
            normal = f'"{p.default}"' if isinstance(p.default, str) else repr(p.default)
            rows.append((p.name + "=", "normally " + code(normal)))
    body = "\n".join(f"{code(name)} & {what}\\\\" for name, what in rows)
    return ("\\wordfield{Options}{}\\par\\nopagebreak\n\\begin{optiontable}\n" + body + "\n\\end{optiontable}")


def _doc_text(doc: str) -> str:
    """A docstring as LaTeX: paragraphs of prose, with lines that are only code set as code."""
    paragraphs = []
    for block in doc.split("\n\n"):
        pieces, prose = [], []
        for line in block.splitlines():
            stripped = line.strip()
            columns = re.search(r"\S( {3,})\S", stripped)
            if columns and "(with sdata.example" not in stripped:
                # a line set out in two columns: code on the left, what it means on the right
                if prose:
                    pieces.append(rich(" ".join(prose)))
                    prose = []
                left, right = stripped[: columns.start(1)], stripped[columns.end(1):]
                pieces.append(r"\codeline{" + esc(left) + r"}\enspace " + rich(" ".join(right.split())))
                continue
            if re.match(r"^(?:sdata|chart|table|nums|grid|page)\.\w+\(|^\w+ = ", stripped):
                if prose:
                    pieces.append(rich(" ".join(prose)))
                    prose = []
                m = re.match(r"^(.*?)\s*(\(with sdata\.example\(.*\))?$", stripped)
                main, with_table = m.group(1), m.group(2)
                text = r"\codeline{" + esc(main) + "}"
                if with_table:
                    text += " " + rich(with_table)
                pieces.append(text)
            else:
                prose.append(stripped)
        if prose:
            pieces.append(rich(" ".join(prose)))
        paragraphs.append("\\\\\n".join(pieces))
    return "\n\n".join(p for p in paragraphs if p)


def _underneath(key: str) -> str:
    if key in UNDERNEATH:
        return UNDERNEATH[key]
    group = learn._group_of(key)
    if group in GROUP_UNDERNEATH:
        return GROUP_UNDERNEATH[group]
    raise KeyError(f'Say which library "{key}" uses: add it to UNDERNEATH in docs/make_manual.py')


def _word_entry(key: str, aliases: dict) -> str:
    title = learn._call_form(key)
    out = [f"\\wordentry{{{esc(title)}}}{{{label(key)}}}"]
    if key in aliases:
        start = {"table": "table", "numbers": "nums", "grid": "grid"}.get(learn._kind(key), "chart")
        other = ", ".join(code(f"{start}.{a}()") for a in aliases[key])
        out.append(f"\\wordfield{{Also written}}{{{other}}}")
    if key in learn.OPTIONS:
        opt = learn.OPTIONS[key]
        out.append(rich(opt["about"]))
        out.append("")
        out.append(rich(opt["text"]))
        out.append("")
        works = ", ".join(f"\\hyperref[{label(w)}]{{{code(f'chart.{w}()')}}}" for w in opt["words"])
        out.append(f"\\wordfield{{Works in}}{{{works}}}")
    else:
        out.append(_doc_text(inspect.getdoc(learn._function(key)) or ""))
        out.append("")
        out.append(_options_table(key))
    if key in learn.EXTRA_NOTES:
        out.append("")
        out.append(r"\wordfield{Good to know}{" + rich(learn.EXTRA_NOTES[key]) + "}")
    out.append("")
    out.append(r"\wordfield{Underneath}{" + rich(_underneath(key)) + "}")
    table = learn._example_table(key)
    where = f"(it uses the practice table {code(table)})" if table else ""
    out.append(r"\wordfield{Example you can run}{" + where + "}\\par\\nopagebreak")
    out.append(listing(learn.HELP_EXAMPLES[key]))
    return "\n".join(out)


def _family(keys: list[str]) -> str | None:
    """Which chapter a group belongs to. A group with table words is a table group even when it has
    a chart word too (survival_curves); a group of sdata words only (finding and solving, planning
    a study) stays in the chapter before it, so no chapter starts twice."""
    kinds = {learn._kind(k) for k in keys} - {"sdata"}
    if not kinds:
        return "help" if "sdata.help" in keys else None
    if "table" in kinds:
        return "table"
    if kinds & {"chart", "option"}:
        return "chart"
    return "numbers"


def _dictionary(aliases: dict) -> str:
    out = []
    current = None
    for group, keys in learn.GROUPS:
        family = _family(keys)
        if family is not None and family != current:
            current = family
            title, intro = FAMILY_TITLES[family]
            out.append(f"\\unchapter{{{title}}}{{family:{family}}}")
            out.append(intro)
        out.append(f"\\unsection{{{esc(group)}}}{{group:{esc(group.replace(' ', '-').replace(':', '').replace(',', ''))}}}")
        for key in keys:
            out.append(_word_entry(key, aliases))
    return "\n".join(out)


def _solutions() -> str:
    out = [r"\unchapter{Solutions to the tasks}{solutions}",
           "One worked solution for each ``Try it yourself'' task. There is often more than one good "
           "answer: if yours runs and answers the question, it is right. Each solution runs on its own "
           "and was run to make the chart and output shown. In Python, "
           + code("sdata.lesson(3, solutions=True)") + " prints the solutions of lesson 3."]
    for number, les in enumerate(learn.LESSONS, start=1):
        out.append(f"\\unsection{{Lesson {number}: {esc(les['title'])}}}{{solutions:{number}}}")
        for task_no, (task, (source, explain)) in enumerate(zip(les["task"], learn.lesson_solutions(number)), start=1):
            out.append(f"\\solutiontask{{{task_no}}}{{{rich(task)}}}")
            out.append(listing(source))
            prefix = _solution_prefix(number, task_no)
            out.append(output_block(prefix))
            fig = figure_block(prefix, source, f"Lesson {number}, task {task_no}:", 0.52, 0.30, numbered=False)
            if fig:
                out.append(fig)
            out.append(f"\\solutionanswer{{{rich(explain)}}}")
        out.append(f"\\par\\noindent{{\\small\\sffamily\\hyperref[lesson:{number}]{{Back to lesson {number}}}}}")
    return "\n".join(out)


def _index(aliases: dict) -> str:
    entries = []
    for key in learn._all_keys():
        entries.append((learn._word(key).lower(), learn._call_form(key), key))
        start = {"table": "table", "numbers": "nums", "grid": "grid"}.get(learn._kind(key), "chart")
        for alias in aliases.get(key, []):
            entries.append((alias.lower(), f"{start}.{alias}()", key))
    entries.sort(key=lambda e: (e[0].strip("="), e[1]))
    lines = [r"\unchapter{Index of words}{index}",
             "Every word and option, A to Z, with the page of its dictionary entry. Other spellings of a "
             "word (such as " + code("chart.bar()") + " for " + code("chart.bars()") + ") point to the same entry.",
             r"\begin{multicols}{2}\small\raggedright\setlength{\parskip}{0.1em}"]
    for _, shown, key in entries:
        lines.append(f"\\indexline{{{label(key)}}}{{{esc(shown)}}}")
    lines.append(r"\end{multicols}")
    return "\n".join(lines)


def in_words(number: int) -> str:
    """A count in words ("twenty"), so that hand-written sentences follow the code."""
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if number < 20:
        return ones[number]
    if number < 100:
        return tens[number // 10] + ("" if number % 10 == 0 else "-" + ones[number % 10])
    return str(number)


def make() -> str:
    """The whole manual as LaTeX text. Runs nothing, so it is quick and always the same."""
    aliases = learn._aliases()
    for key in learn._all_keys():  # every word must say which library it uses
        _underneath(key)
    body = [
        _preamble(),
        r"\begin{document}",
        r"\frontmatter",
        _title_page(),
        r"\tableofcontents",
        r"\mainmatter",
        r"\part{Getting started}\label{part:start}",
        GETTING_STARTED.strip().replace("@LESSON_COUNT@", in_words(len(learn.LESSONS)).capitalize()),
        _practice_tables(),
        r"\part{Lessons}\label{part:lessons}",
        r"\lessonchapters",
    ]
    body += [_lesson(n, aliases) for n in range(1, len(learn.LESSONS) + 1)]
    body += [
        r"\otherchapters",
        r"\part{Dictionary of words}\label{part:dictionary}",
        r"\unchapter{How to read the dictionary}{dictionary:how}",
        "Each entry gives the word as you type it, what it does (in the words of "
        + code('sdata.help("word")') + "), the options you can put in its brackets and what they "
        "normally are, the library that does the work underneath, and an example you can run as it is. "
        "The same entries are in Python: " + code('sdata.help("bars")') + ". The words come in the same "
        "groups as " + code("sdata.words()") + "; the \\hyperref[index]{index of words} at the end of the "
        "book lists them A to Z.",
        _dictionary(aliases),
        r"\part{Data science ideas}\label{part:ideas}",
        r"\input{ideas}",
        r"\part{Solutions}\label{part:solutions}",
        _solutions(),
        r"\backmatter",
        _index(aliases),
        r"\end{document}",
    ]
    return "\n\n".join(body) + "\n"


# ---------------------------------------------------------------- running the examples
def _work_folder() -> Path:
    """Scratch space for the build: F:\\claude_work if it exists (keeps the project folder
    small), else the system's temporary folder."""
    base = Path(r"F:\claude_work\scratch\simplydata")
    if not base.parent.exists():
        base = Path(tempfile.gettempdir()) / "simplydata_manual"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _as_pdf(source: str) -> str:
    """Save pictures as PDF (sharp at any size) instead of PNG; animations stay .gif."""
    def swap(m):
        stem, _, kind = m.group(1).rpartition(".")
        if kind.lower() in IMAGE_TYPES and kind.lower() != "gif":
            return f'.save("{stem}.pdf"'
        return m.group(0)
    return _SAVE.sub(swap, source)


def _last_frame(gif: Path, png: Path) -> None:
    from PIL import Image

    with Image.open(gif) as im:
        im.seek(im.n_frames - 1)
        im.convert("RGB").save(png, optimize=True)


def _clean_output(text: str) -> str:
    lines = [line.rstrip().replace("\t", "    ") for line in text.strip("\n").splitlines()]
    if len(lines) > MAX_OUTPUT_LINES:
        cut = len(lines) - (MAX_OUTPUT_LINES - 2)
        lines = lines[: MAX_OUTPUT_LINES - 2] + ["", f"... ({cut} more lines)"]
    return "\n".join(lines) + "\n" if lines else ""


def run_examples() -> None:
    """Run every lesson example and solution; keep their pictures and printed output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["SOURCE_DATE_EPOCH"] = "0"  # no dates in the PDF pictures: they only change with the chart
    for folder in (FIGURES, OUTPUT):  # empty them (the folders stay: a synced folder may be locked)
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            old.unlink()
    work = _work_folder() / "examples"
    if work.exists():
        shutil.rmtree(work)
    start = Path.cwd()
    for prefix, source in jobs():
        folder = work / prefix
        folder.mkdir(parents=True)
        os.chdir(folder)
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile(_as_pdf(source), f"<{prefix}>", "exec"), {"__name__": "__main__"})
        finally:
            plt.close("all")
            os.chdir(start)
        text = _clean_output(printed.getvalue())
        if text:
            (OUTPUT / f"{prefix}.txt").write_text(text, encoding="utf-8")
        for (stem, kind), target in zip(saved_images(source), figure_files(prefix, source)):
            if kind == "gif":
                _last_frame(folder / f"{stem}.gif", MANUAL / target)
            else:
                shutil.copyfile(folder / f"{stem}.pdf", MANUAL / target)
        print(f"  ran {prefix}")


def latex_folder() -> Path:
    """Where LaTeX writes its build files (.aux, .log ...): SIMPLYDATA_LATEX_DIR, else the scratch space."""
    build = Path(os.environ.get("SIMPLYDATA_LATEX_DIR") or _work_folder() / "latex")
    build.mkdir(parents=True, exist_ok=True)
    return build


def compile_pdf() -> Path:
    build = latex_folder()
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
    for run in range(3):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", TEX.name],
            cwd=MANUAL, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            log = (build / "simplydata_manual.log").read_text(encoding="utf-8", errors="replace")
            print(log[-4000:])
            raise SystemExit(f"pdflatex failed (run {run + 1}); the full log is in {build}")
    shutil.copyfile(build / "simplydata_manual.pdf", PDF)
    return build


def main(argv: list[str]) -> None:
    MANUAL.mkdir(exist_ok=True)
    TEX.write_text(make(), encoding="utf-8")
    print(f"Wrote {TEX}")
    if "--tex" in argv:
        return
    if "--no-run" not in argv:  # --no-run: compile with the pictures and output already there
        print("Running every lesson example and solution ...")
        run_examples()
    print("Compiling the PDF ...")
    build = compile_pdf()
    print(f"Wrote {PDF}")
    import make_dictionary  # the dictionary takes its page numbers from the manual just compiled

    make_dictionary.build(build)
    import make_30_days  # and so does the 30-day guide

    make_30_days.build(build)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1:])
