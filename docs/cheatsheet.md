# simplydata cheat sheet

Every word on one page. Table words are steps on a table, for looking at data, cleaning
it and summing it up; chart words are steps on a chart:

```python
import simplydata as sdata
table = sdata.table(sdata.example("marks"))   # or sdata.table("my_file.xlsx")
table.keep_rows("marks", above=50)
table.average_of("marks", for_each="class")   # says the answer in a sentence
table.save("good_marks.xlsx")
chart = table.chart()                   # or sdata.chart(table), sdata.chart("my_file.xlsx")
chart.bars(x="class", y="marks", error="standard error")
chart.title_top("Good marks")
chart.save("marks.png")
```

Number words are steps on a list (or a grid) of numbers:

```python
nums = sdata.numbers([12, 15, 9, 20, 14])
bigger = nums * 2                       # symbols make new numbers
nums.keep_numbers(above=10)             # words change the numbers (undo() takes it back)
nums.average()
grid = sdata.number_grid([[3, 5, 2], [8, 6, 4]])
```

In Python, `sdata.help("bars")` explains any word, `sdata.words()` prints this list and
`sdata.lesson(1)` starts the one-day course (lessons 9 to 12 are about tables, lesson 13
about numbers, lesson 14 about statistical tests, lesson 15 puts it all together, and lessons 16 to 20
go further: models, patterns, survival, fitting curves, finding and solving, and distributions).

Every example runs as written on the practice table in the *Table* column:
for example `weather = sdata.example("weather")` then `chart = sdata.chart(weather)`,
or `table = sdata.table(sdata.example("weather"))` for table words.
Number words use `nums = sdata.numbers([12, 15, 9, 20, 14])` and grid words
`grid = sdata.number_grid([[3, 5, 2], [8, 6, 4]])`; `sdata.help("word")` shows each example ready to run.
Steps such as `fit_line` or `title_top` go after a chart step such as `chart.dots(...)`,
and `a`, `b` in `sdata.side_by_side(a, b)` are charts you made first.

*This file is made by `docs/make_cheatsheet.py`. Do not edit it by hand.*

## Start, show and save

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.chart` | `sdata.chart(sdata.example("marks"))` | marks | Start a chart. |
| `sdata.example` | `sdata.example("marks")` | marks | A small made-up table to practise with. |
| `show` | `chart.show()` | marks | Show the chart in a window. |
| `save` | `chart.save("marks.png")` | marks | Save the chart. |

## Chart types

| Word | Example | Table | What it does |
|---|---|---|---|
| `bars` (also `bar`) | `chart.bars(x="name", y="marks")` | marks | Bars. |
| `lines` (also `line`) | `chart.lines(x="month", y="temperature", colour="city")` | weather | Lines. |
| `dots` (also `scatter`) | `chart.dots(x="hours", y="marks")` | marks | Dots (a scatter chart). |
| `area` | `chart.area(x="year", y="population", colour="country")` | growth | A line with the space under it coloured in. |
| `histogram` | `chart.histogram(x="marks", bins=10)` | marks | Bars that count how many values fall in each range. |
| `boxes` (also `box`) | `chart.boxes(x="class", y="marks")` | marks | Boxes that show the middle half of the values and the median. |
| `violins` (also `violin`) | `chart.violins(x="class", y="marks")` | marks | Shapes that are wide where many values are. |
| `heatmap` | `chart.heatmap(x="month", y="city", value="rain", numbers=True)` | weather | A grid of coloured squares, one colour for each number. |
| `pie` | `chart.pie(labels="fruit", values="count")` | fruit | A circle cut into slices, one slice per group. |
| `polar` | `chart.polar(angle="wind_direction", distance="wind_speed", colour="city")` | weather | Dots on a round chart: how far round, and how far out. |
| `contour` | `chart.contour(x="east", y="north", z="height")` | island | A map where each colour band joins places with the same height. |
| `surface` | `chart.surface(x="east", y="north", z="height")` | island | A 3D landscape: z= is how high the ground is. |
| `dots_3d` | `chart.dots_3d(x="hours", y="height", z="marks", colour="class")` | marks | Dots floating in a 3D box. |
| `band` | `chart.band(x="month", between=["lowest", "highest"])` | weather | A shaded band between a low and a high column, along x. |
| `empty_cells` | `chart.empty_cells()` | shop | A map of the empty cells: one row of the chart for each row of the table, one strip for each column, with empty cells coloured. |

## Statistics

| Word | Example | Table | What it does |
|---|---|---|---|
| `show=` | `chart.bars(x="class", y="marks", show="median")` | marks | What each bar or point on a line stands for, when a group has many rows. |
| `error=` | `chart.bars(x="class", y="marks", error="standard deviation")` | marks | Error bars on top of averages: how spread out, or how sure, the values are. |
| `error_bars` | `chart.error_bars("standard error")` | marks | Error bars on dots or lines. |
| `fit_line` | `chart.fit_line()` | marks | A straight line that fits the dots best (linear regression). |
| `fit_curve` | `chart.fit_curve(degree=2)` | weather | A curve that fits the dots best. |
| `fit_formula` | `chart.fit_formula("population = a * exp(b * (year - 2000))")` | growth | The curve of a formula fitted to the dots. |
| `fit_distribution` | `chart.fit_distribution("normal")` | marks | The best-fitting distribution's curve over a histogram or density chart. |
| `curve` | `chart.curve("y = 100 * exp(-0.3 * x)", between=[0, 10])` | marks | Draw a formula, or a distribution, as a curve. |
| `smooth_line` | `chart.smooth_line()` | weather | A smooth line that follows the dots (LOWESS). |
| `density` | `chart.density(x="marks", colour="class")` | marks | A smooth curve that is high where many values are. |
| `count_bars` | `chart.count_bars(x="fruit", colour="shop")` | fruit | Bars that count how many rows each group has. |
| `all_dots` | `chart.all_dots(x="class", y="marks")` | marks | Every value as a dot, spread out so no dot hides another. |
| `side_histograms` | `chart.side_histograms()` | marks | Small histograms along the top and right edges of a dots chart. |
| `qq_dots` | `chart.qq_dots("marks")` | marks | Dots that lie on a straight line when the values follow a normal (bell) curve. |
| `cumulative_share` | `chart.cumulative_share(x="marks", colour="class")` | marks | A line that climbs from 0 to 1: the share of values at or below each x. |
| `sdata.pairs` | `sdata.pairs(sdata.example("marks"), colour="class")` | marks | Every number column against every other. |

## Lines and shading

| Word | Example | Table | What it does |
|---|---|---|---|
| `line_at_x` | `chart.line_at_x(2010, label="2010")` | growth | A straight up-and-down line at one x value. |
| `line_at_y` | `chart.line_at_y(50, label="pass mark")` | marks | A straight side-to-side line at one y value. |
| `line_with_slope` | `chart.line_with_slope(1, through=(0, 0))` | marks | A straight line with a given slope, right across the chart. |
| `shade_between_x` | `chart.shade_between_x(6, 8)` | weather | Shade the band between two x values, behind the data. |
| `shade_between_y` | `chart.shade_between_y(40, 60)` | marks | Shade the band between two y values, behind the data. |

## Words on the chart

| Word | Example | Table | What it does |
|---|---|---|---|
| `title_top` | `chart.title_top("My class marks")` | marks | A title above the chart. |
| `caption_bottom` | `chart.caption_bottom("Data from 2026")` | marks | A small line of text under the whole chart. |
| `label_x` | `chart.label_x("Student")` | marks | The name under the x-axis. |
| `label_y` | `chart.label_y("Marks")` | marks | The name beside the y-axis. |
| `note` | `chart.note("Biggest", x=2024, y=49)` | growth | A note on the chart at a place x, y. |
| `numbers_on_bars` | `chart.numbers_on_bars()` | fruit | Write the value of each bar on it. |
| `names_on_dots` | `chart.names_on_dots("name")` | marks | Write a name next to each dot. |
| `font_size` | `chart.font_size(14)` | marks | Make all the text bigger or smaller. |
| `font_name` | `chart.font_name("DejaVu Serif")` | marks | The font for all the text on the chart. |

## Axes

| Word | Example | Table | What it does |
|---|---|---|---|
| `range_x` | `chart.range_x(0, 10)` | marks | From where to where the x-axis goes. |
| `range_y` | `chart.range_y(0, 100)` | marks | From where to where the y-axis goes. |
| `log_scale_x` | `chart.log_scale_x()` | growth | A log scale on the x-axis: 1, 10, 100, 1000 are evenly spaced. |
| `log_scale_y` | `chart.log_scale_y()` | growth | A log scale on the y-axis: 1, 10, 100, 1000 are evenly spaced. |
| `ticks_x` | `chart.ticks_x(1, 4, 7, 10)` | weather | Where the marks on the x-axis go. |
| `ticks_y` | `chart.ticks_y(0, 50, 100)` | marks | Where the marks on the y-axis go. |
| `rotate_labels_x` | `chart.rotate_labels_x()` | marks | Turn the names under the x-axis so long ones do not bump into each other. |
| `rotate_labels_y` | `chart.rotate_labels_y(30)` | fruit | Turn the names beside the y-axis. |
| `tick_format_x` | `chart.tick_format_x("thousands")` | growth | How the numbers or dates under the x-axis are written. |
| `tick_format_y` | `chart.tick_format_y("percent")` | growth | How the numbers beside the y-axis are written. |
| `second_y` | `chart.second_y("rain", drawn_as="bars", label="Rain (mm)")` | weather | A second y-axis on the right, for a column measured in other units. |
| `flip_x` | `chart.flip_x()` | marks | Turn the x-axis around, so numbers grow from right to left. |
| `flip_y` | `chart.flip_y()` | island | Turn the y-axis upside down, for example for depth going down. |

## Colours and look

| Word | Example | Table | What it does |
|---|---|---|---|
| `colour` (also `color`) | `chart.colour("blue")` | marks | One colour for the whole chart. |
| `colours` (also `colors`) | `chart.colours("blue", "orange", "green")` | marks | Your own colours for the groups, in order. |
| `colour_scale` (also `color_scale`) | `chart.colour_scale("magma")` | weather | The colours for numbers on a heatmap, contour map, surface or dots coloured by a number column. |
| `style` | `chart.style("school")` | marks | The overall look. |
| `size` | `chart.size(width=17, height=12, unit="cm")` | weather | The size of the chart. |
| `show_grid` | `chart.show_grid()` | marks | Grid lines across the chart in both directions. |
| `hide_grid` | `chart.hide_grid()` | marks | No grid lines. |
| `legend_top` | `chart.legend_top()` | weather | The legend above the chart, in one row. |
| `legend_bottom` | `chart.legend_bottom()` | weather | The legend below the chart, in one row under the x-axis name. |
| `legend_left` | `chart.legend_left()` | weather | The legend to the left of the chart. |
| `legend_right` | `chart.legend_right()` | weather | The legend to the right of the chart, outside the plot. |
| `hide_legend` | `chart.hide_legend()` | weather | No legend. |
| `show_legend` | `chart.show_legend()` | weather | The legend in its usual place, for example after hide_legend(). |
| `label_legend` | `chart.label_legend("Class")` | marks | The name above the legend. |
| `label_colour_bar` (also `label_color_bar`) | `chart.label_colour_bar("Height (m)")` | island | The name beside the colour bar of a heatmap or contour. |

## Order of groups

| Word | Example | Table | What it does |
|---|---|---|---|
| `order_x` | `chart.order_x("C", "B", "A")` | marks | Put the groups along the x-axis in your order. |
| `order_y` | `chart.order_y("C", "B", "A")` | marks | Put the groups along the y-axis in your order, the first at the top. |
| `order_legend` | `chart.order_legend("Science", "Maths", "English")` | marks | Put the colour groups in your order: in the legend, the colours and side by side. |

## Several charts

| Word | Example | Table | What it does |
|---|---|---|---|
| `one_chart_per` | `chart.one_chart_per("class", per_row=3)` | marks | One small chart for each group. |
| `sdata.side_by_side` | `sdata.side_by_side(a, b)` | marks | Charts next to each other. |
| `sdata.one_above_other` | `sdata.one_above_other(a, b)` | weather | Charts on top of each other. |
| `sdata.grid` | `sdata.grid(a, b, c, d, per_row=2)` | marks | Charts in rows and columns. |
| `zoom_box` | `chart.zoom_box(x_between=[2, 4], y_between=[40, 60])` | marks | A small box showing part of the chart close up. |
| `animate` | `chart.animate(over="year")` | growth | A moving chart with one picture for each value of a column. |

## Tables: start and look

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.table` | `sdata.table(sdata.example("marks"))` | marks | Start a table. |
| `show_first_rows` | `table.show_first_rows(5)` | marks | Show the first rows. |
| `show_last_rows` | `table.show_last_rows(5)` | marks | Show the last rows. |
| `show` | `table.show()` | marks | Show the table, every row and every column (or only the first rows=). |
| `column_names` | `table.column_names()` | marks | The names of all columns. |
| `describe_columns` | `table.describe_columns()` | marks | Describe every column: its kind, empty cells, smallest, biggest and average. |
| `column` | `table.column("marks")` | marks | One column on its own: a column of numbers becomes sdata.numbers, any other a plain list. |

## Choose columns

| Word | Example | Table | What it does |
|---|---|---|---|
| `keep_columns` | `table.keep_columns("name", "marks")` | marks | Keep only these columns, in the order you list them. |
| `remove_columns` | `table.remove_columns("height")` | marks | Remove these columns. |
| `rename_column` | `table.rename_column("marks", to="score")` | marks | Give a column a new name. |

## Choose rows

| Word | Example | Table | What it does |
|---|---|---|---|
| `keep_rows` | `table.keep_rows("marks", above=50)` | marks | Keep only the rows that match a condition. |
| `remove_rows` | `table.remove_rows("subject", equal_to="English")` | marks | Remove the rows that match a condition (the opposite of keep_rows). |
| `keep_first_rows` | `table.keep_first_rows(10)` | marks | Keep only the first rows. |
| `keep_last_rows` | `table.keep_last_rows(10)` | marks | Keep only the last rows. |
| `keep_random_rows` | `table.keep_random_rows(10, seed=1)` | marks | Keep some rows picked at random; give a seed to get the same rows every time. |
| `remove_duplicate_rows` | `table.remove_duplicate_rows(columns=["name"])` | marks | Remove rows that repeat an earlier row, or that repeat it in the columns you name. |
| `keep_biggest_rows` | `table.keep_biggest_rows(3, by="marks", for_each="subject")` | marks | Keep the n rows with the biggest values in a column (also for each group). |
| `keep_smallest_rows` | `table.keep_smallest_rows(3, by="marks", for_each="subject")` | marks | Keep the n rows with the smallest values in a column (also for each group). |
| `show_rows_with_outliers` | `table.show_rows_with_outliers("marks", for_each="subject")` | marks | Show the rows whose value in a column is an outlier, and say why (the table does not change). |
| `remove_rows_with_outliers` | `table.remove_rows_with_outliers("marks", for_each="subject")` | marks | Remove the rows whose value in a column is an outlier, and say how many and why. |

## Empty cells

| Word | Example | Table | What it does |
|---|---|---|---|
| `count_empty_cells` | `table.count_empty_cells()` | shop | Show how many empty cells each column has. |
| `remove_rows_with_empty_cells` | `table.remove_rows_with_empty_cells("sales")` | shop | Remove rows with an empty cell (in any column, or only in the column you name). |
| `fill_empty_cells` | `table.fill_empty_cells("customers", with_the="median")` | shop | Fill the empty cells of a column with a value, or with the average, median, value above or value below. |
| `add_is_empty_column` | `table.add_is_empty_column("sales_was_empty", from_column="sales")` | shop | Add a true/false column: True where a cell of from_column is empty. |

## Fix columns

| Word | Example | Table | What it does |
|---|---|---|---|
| `make_numbers` | `table.make_numbers("marks")` | marks | Turn a column into numbers; cells that are not numbers become empty. |
| `make_text` | `table.make_text("month_number")` | weather | Turn a column into text. |
| `make_dates` | `table.make_dates("year")` | growth | Turn a column into dates; cells that are not dates become empty. |
| `make_lowercase` | `table.make_lowercase("name")` | marks | Write all the text in a column in small letters. |
| `make_uppercase` | `table.make_uppercase("name")` | marks | Write all the text in a column in capital letters. |
| `remove_extra_spaces` | `table.remove_extra_spaces("name")` | marks | Remove spaces at the start and end of the text, and double spaces inside it. |
| `replace_values` | `table.replace_values("subject", "Maths", with_value="Math")` | marks | Replace a value (or a list of values) in a column; with_value=None makes those cells empty. |
| `round_to` | `table.round_to("hours", 0)` | marks | Round numbers to this many decimal places: in one column, a list of columns, or every number column. |
| `change_column` | `table.change_column("marks", "marks / 10")` | marks | Change the values of a column in place, with the same ways as add_column(). |

## Sort and new columns

| Word | Example | Table | What it does |
|---|---|---|---|
| `sort_by` | `table.sort_by("marks", biggest_first=True)` | marks | Put the rows in order of one or more columns; empty cells go last. |
| `add_column` | `table.add_column("study_and_marks", by_adding=["hours", "marks"])` | marks | Add a new column, worked out from other columns in plain words or with a formula. |
| `add_group_column` | `table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])` | marks | Add a column that puts each row into a group, like turning marks into grades. |
| `add_rank_column` | `table.add_rank_column("place", from_column="marks")` | marks | Add a column with each row's place, where 1 is the best. |

## Dates and text

| Word | Example | Table | What it does |
|---|---|---|---|
| `add_date_part_column` | `table.add_date_part_column("weekday", from_column="date", part="weekday name")` | shop | Add a column with one part of each date, such as its year, month or weekday. |
| `add_text_part_column` | `table.add_text_part_column("initial", from_column="name", first_characters=1)` | marks | Add a column with one part of each text, such as its first letters or the text before a mark. |
| `split_column` | `table.split_column("shop", separator=" ", into=["shop_kind", "shop_word"])` | fruit | Split a text column into several columns at a separator, such as a comma or a space. |
| `replace_text` | `table.replace_text("shop", " shop", with_value="")` | fruit | Replace a piece of text inside the cells of a text column, wherever it appears. |

## Row after row

| Word | Example | Table | What it does |
|---|---|---|---|
| `add_rolling_column` | `table.add_rolling_column("rain_3_months", from_column="rain", window=3, show="total",
                           for_each="city", sorted_by="month_number")` | weather | Add a rolling (moving) summary: for each row, the average of this row and the rows just before it. |
| `add_previous_value_column` | `table.add_previous_value_column("last_month", from_column="temperature", for_each="city",
                                  sorted_by="month_number")` | weather | Add a column with the value from the row before (or rows_back= rows before). |
| `add_change_column` | `table.add_change_column("growth", from_column="population", in_percent=True,
                          for_each="country", sorted_by="year")` | growth | Add a column with the change from the row before: this value minus the previous one. |

## Summaries

| Word | Example | Table | What it does |
|---|---|---|---|
| `count_rows` | `table.count_rows(for_each="class")` | marks | How many rows the table has, or how many in each group. |
| `count_each` | `table.count_each("fruit")` | fruit | Each different value in a column, how many times it appears, and its percent. |
| `average_of` | `table.average_of("marks", for_each="class")` | marks | The average (mean) of a column; empty cells are skipped. |
| `median_of` | `table.median_of("marks", for_each="class")` | marks | The middle number of a column once it is sorted; empty cells are skipped. |
| `total_of` | `table.total_of("count", for_each="shop")` | fruit | All the numbers in a column added up; empty cells are skipped. |
| `smallest_of` | `table.smallest_of("temperature", for_each="city")` | weather | The smallest number in a column; empty cells are skipped. |
| `biggest_of` | `table.biggest_of("temperature", for_each="city")` | weather | The biggest number in a column; empty cells are skipped. |
| `standard_deviation_of` | `table.standard_deviation_of("marks", for_each="class")` | marks | How spread out the numbers are: the sample standard deviation (divide by n - 1). |
| `summary_of` | `table.summary_of("marks", for_each="class", show=["average", "standard deviation", "count"])` | marks | Several measures of one or more columns at once, in one tidy table. |
| `percent_of` | `table.percent_of("marks", above=60, for_each="class")` | marks | What percent of the rows meet a condition, or each group's share. |
| `count_different_values` | `table.count_different_values("name", for_each="class")` | marks | How many different values a column has, or how many in each group. |
| `percentile_of` | `table.percentile_of("marks", 90, for_each="class")` | marks | The value below which a given percent of the numbers lie; empty cells are skipped. |
| `most_common_of` | `table.most_common_of("days_absent", for_each="school")` | school | The value that turns up most often (the mode), for numbers or text; empty cells are skipped. |
| `correlation_between` | `table.correlation_between("hours", "marks")` | marks | How strongly columns of numbers go up or down together, from -1 to 1. |

## Statistical tests

| Word | Example | Table | What it does |
|---|---|---|---|
| `compare_averages` | `table.compare_averages("marks", for_each="class")` | marks | Are the averages of the groups really different, or could it be chance? |
| `compare_each_pair` | `table.compare_each_pair("marks", for_each="class")` | marks | Which pairs of groups differ? Every pair is compared, with Holm's correction. |
| `compare_before_after` | `table.compare_before_after("Maths", "Science")` | marks | Did the values change from one column to the other, row by row, more than chance? |
| `compare_average_to` | `table.compare_average_to("marks", 60)` | marks | Is the average of a column different from a number you choose, or could it be chance? |
| `compare_counts` | `table.compare_counts("result", for_each="class")` | marks | Does how often each value turns up differ between groups, more than chance? |
| `compare_shape_to` | `table.compare_shape_to("marks", "normal", for_each="class")` | marks | Do the values look bell-shaped (normal), or are they too far from it to be chance? |
| `add_adjusted_p_value_column` | `table.add_adjusted_p_value_column("q", method="benjamini-hochberg")` | marks | Adjust p-values for testing many things at once, in a new column. |
| `compare_spreads` | `table.compare_spreads("marks", for_each="class")` | marks | Do the groups differ in how spread out their values are, more than chance? |
| `compare_counts_to` | `table.compare_counts_to("breakfast", expected={"yes": 0.6, "no": 0.4})` | school | Does how often each value turns up match the shares you expect, more than chance? |
| `compare_percent_to` | `table.compare_percent_to("passed", value="yes", percent=50)` | school | Is the percent of rows with one value different from a percent you choose, or could it be chance? |

## Fitting

| Word | Example | Table | What it does |
|---|---|---|---|
| `fit_line` | `table.fit_line(x="hours", y="marks")` | marks | Fit the straight line y = slope * x + intercept that best follows the points. |
| `fit_curve` | `table.fit_curve(x="hours", y="marks", degree=2)` | marks | Fit a curve y = b0 + b1 * x + b2 * x**2 ... that best follows the points. |
| `fit_formula` | `table.fit_formula("population = a * exp(b * (year - 2000))")` | growth | Find the values in a formula that make it follow the data best. |

## Finding and solving

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.find_smallest` | `sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])` |  | Find where a formula is smallest, and its value there. |
| `sdata.find_biggest` | `sdata.find_biggest("3*x + 2*y", rules=["x + y <= 4", "x + 3*y <= 6", "x >= 0", "y >= 0"])` |  | Find where a formula is biggest, and its value there. |
| `sdata.solve` | `sdata.solve("x**3 - 2*x = 5", between=[0, 3])` |  | Find the values that make an equation true. |

## Values between points

| Word | Example | Table | What it does |
|---|---|---|---|
| `values_at` | `table.values_at("population", at={"year": [2001, 2003, 2011]}, for_each="country")` | growth | Estimate a column at places between the known points, by joining the dots. |
| `map_of` | `table.map_of("height", across=["east", "north"], steps=100)` | island | Estimate a column at evenly spaced places across one or more columns (a map), ready for a contour or heatmap. |

## Distributions

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.distribution` | `sdata.distribution("normal", average=50, standard_deviation=10)` |  | Make a distribution to ask chances, percentiles and random numbers of. |
| `fit_distribution` | `table.fit_distribution("sales", "lognormal")` | shop | Fit a distribution to a column by maximum likelihood; gives back the fitted distribution. |
| `compare_distributions` | `table.compare_distributions("sales")` | shop | Fit several distributions to a column and rank them by AIC (smaller is better). |
| `fit_distribution` | `nums.fit_distribution("gamma")` |  | Fit a distribution to the numbers by maximum likelihood; gives back the fitted distribution. |
| `chance` | `dist.chance(above=80)` |  | The chance of a value in a range, as a share from 0 to 1 (printed as a percent). |
| `density_at` | `dist.density_at(60)` |  | The height of the density curve at a value (continuous distributions). |
| `percentile` | `dist.percentile(95)` |  | The value that this percent of the distribution is below: percentile(95). |
| `cut_off` | `dist.cut_off(above=0.025)` |  | The value with this chance above it (or below it): critical values. |
| `middle_range` | `dist.middle_range(95)` |  | The range that holds the middle percent of the distribution, as Numbers [low, high]. |
| `average` | `dist.average()` |  | The average (mean) of the distribution. |
| `median` | `dist.median()` |  | The median: half of the distribution is below it. |
| `standard_deviation` | `dist.standard_deviation()` |  | The standard deviation of the distribution. |
| `variance` | `dist.variance()` |  | The variance (the standard deviation squared). |
| `summary` | `dist.summary()` |  | A one-row table: average, median, standard deviation, variance, skewness, excess kurtosis, entropy and the lowest and highest possible values. |
| `random_numbers` | `dist.random_numbers(how_many=100, seed=1)` |  | Numbers drawn at random from the distribution; the same seed gives the same numbers. |
| `chart` | `dist.chart()` |  | A chart of the distribution: its density curve (or bars for counts). |
| `table` | `dist.table()` |  | A one-row table: the distribution, its parameters and, for a fit, the standard errors, log-likelihood, AIC and BIC. |
| `parameters` | `dist.parameters()` |  | The parameters as a dictionary of plain names: {"average": 50, "standard_deviation": 10}. |

## Models

| Word | Example | Table | What it does |
|---|---|---|---|
| `explain` | `table.explain("marks", using=["hours", "breakfast"])` | school | Explain one column from several others with a model, and say how big each effect is. |
| `add_predicted_column` | `table.add_predicted_column("marks", using=["hours", "breakfast"])` | school | Add a column with the value the model predicts for each row. |

## Patterns across columns

| Word | Example | Table | What it does |
|---|---|---|---|
| `add_principal_components` | `table.add_principal_components(how_many=2)` | meadow | Add principal components: a few new columns that hold most of the spread of many columns. |
| `principal_components_of` | `table.principal_components_of()` | meadow | A table of the principal components: how much spread each explains and the weight of each column. |
| `add_clusters` | `table.add_clusters(how_many=3)` | meadow | Sort the rows into groups of similar rows (clusters), in a new column. |
| `distances_between_rows` | `table.distances_between_rows(names="plot")` | meadow | How different each row is from every other row: a square table of distances. |
| `add_nmds` | `table.add_nmds()` | meadow | Draw the rows as points on a map where close points are similar rows (NMDS), in new columns. |
| `compare_profiles` | `table.compare_profiles(for_each="grazed")` | meadow | Do the groups differ in their make-up across many columns (PERMANOVA), or could it be chance? |

## Survival

| Word | Example | Table | What it does |
|---|---|---|---|
| `survival_over_time` | `table.survival_over_time(time="days", event="died", for_each="treatment")` | trial | The share still without the event over time (Kaplan-Meier), with the median time. |
| `compare_survival` | `table.compare_survival(time="days", event="died", for_each="treatment")` | trial | Do the groups differ in the time until the event, or could it be chance? (log-rank test) |
| `explain_survival` | `table.explain_survival(time="days", event="died", using=["treatment", "age"])` | trial | Which columns go with a higher or lower risk of the event? (Cox model, hazard ratios) |
| `survival_curves` | `chart.survival_curves(time="days", event="died", colour="treatment")` | trial | The share still without the event over time (Kaplan-Meier curves). |

## Planning a study

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.how_many_needed` | `sdata.how_many_needed(effect_size=0.5)` |  | How many values (in each group) does a study need to find a real difference? |
| `sdata.chance_to_find` | `sdata.chance_to_find(effect_size=0.5, values_per_group=30)` |  | What chance does a study have to find a real difference of a given size (its power)? |

## Combine and reshape

| Word | Example | Table | What it does |
|---|---|---|---|
| `join_with` | `table.join_with({"name": ["Asha", "Ben"], "age": [12, 13]}, using="name")` | marks | Bring in the columns of another table, matching rows on a column such as a name. |
| `add_rows_from` | `table.add_rows_from(sdata.example("marks"))` | marks | Put the rows of another table under this one; columns are matched by their names. |
| `turn_columns_into_rows` | `table.turn_columns_into_rows(["temperature", "rain"])` | weather | Turn some columns into rows: each value gets its own row, with its old column name beside it. |
| `turn_rows_into_columns` | `table.turn_rows_into_columns(names_from="city", values_from="temperature")` | weather | Turn rows into columns: each different value in names_from= becomes a new column, filled from values_from=. |

## Undo and finish

| Word | Example | Table | What it does |
|---|---|---|---|
| `undo` | `table.undo()` | marks | Take the last step back. |
| `steps` | `table.steps()` | marks | List every step you did to this table, in order. |
| `start_again` | `table.start_again()` | marks | Go back to the table as it was at the start. |
| `save` | `table.save("clean.xlsx")` | marks | Save the table; the end of the file name decides the type. |
| `chart` | `table.chart()` | marks | Start a chart from this table. |
| `as_pandas` | `table.as_pandas()` | marks | Give back the table as a plain pandas table, for other Python tools. |
| `as_polars` | `table.as_polars()` | marks | Give back the table as a plain Polars table, for other Python tools. |

## Numbers: make and look

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.numbers` | `sdata.numbers([3, 5, 2, 8])` |  | Make a list of numbers. |
| `sdata.random_numbers` | `sdata.random_numbers(how_many=10, between=[0, 1], seed=1)` |  | Make random numbers; give seed= to get the same numbers every time. |
| `show` | `nums.show()` |  | Show every number. |
| `show_first_numbers` | `nums.show_first_numbers(5)` |  | Show the first numbers. |
| `show_last_numbers` | `nums.show_last_numbers(5)` |  | Show the last numbers. |
| `number_at` | `nums.number_at(1)` |  | The number at a position, counting from 1: number_at(1) is the first number. |
| `count` | `nums.count()` |  | How many numbers there are (empty ones included). |
| `count_empty_numbers` | `nums.count_empty_numbers()` |  | How many values are empty (missing). |

## Numbers: maths in words

| Word | Example | Table | What it does |
|---|---|---|---|
| `add` | `nums.add(5)` |  | Add a number to every number (or add numbers one by one). |
| `subtract` | `nums.subtract(2)` |  | Take a number away from every number. |
| `times` | `nums.times(2)` |  | Multiply every number by a number. |
| `divide_by` | `nums.divide_by(4)` |  | Divide every number by a number. |
| `power` | `nums.power(2)` |  | Raise every number to a power: power(2) squares, power(3) cubes. |
| `square_root` | `nums.square_root()` |  | The square root of every number: 9 becomes 3. |
| `absolute_value` | `nums.absolute_value()` |  | Every number without its minus sign: -3 becomes 3. |
| `round_to` | `nums.round_to(2)` |  | Round every number to this many decimal places. |
| `log` | `nums.log()` |  | The logarithm of every number: natural (base e) unless you give base=10 or base=2. |
| `exp` | `nums.exp()` |  | e (about 2.718) to the power of every number; it undoes log(). |
| `running_total` | `nums.running_total()` |  | Each number becomes the total so far (the cumulative sum): 1, 2, 3 becomes 1, 3, 6. |

## Numbers: choose, sort and fill

| Word | Example | Table | What it does |
|---|---|---|---|
| `keep_numbers` | `nums.keep_numbers(above=50)` |  | Keep only the numbers that match one condition. |
| `remove_numbers` | `nums.remove_numbers(below=0)` |  | Remove the numbers that match one condition (the opposite of keep_numbers). |
| `keep_first_numbers` | `nums.keep_first_numbers(10)` |  | Keep only the first numbers. |
| `keep_last_numbers` | `nums.keep_last_numbers(10)` |  | Keep only the last numbers. |
| `keep_random_numbers` | `nums.keep_random_numbers(10, seed=1)` |  | Keep some numbers picked at random (they keep their order); a seed gives the same pick every time. |
| `remove_duplicate_numbers` | `nums.remove_duplicate_numbers()` |  | Remove numbers that repeat an earlier number, so each number is left once, in its first place. |
| `remove_empty_numbers` | `nums.remove_empty_numbers()` |  | Remove the empty (missing) values. |
| `fill_empty_numbers` | `nums.fill_empty_numbers(with_value=0)` |  | Fill the empty values with a number, or with the average, median, value above or value below. |
| `sort` | `nums.sort(biggest_first=True)` |  | Put the numbers in order, smallest first; empty values go last. |
| `reverse` | `nums.reverse()` |  | Turn the order round: the last number comes first. |

## Numbers: summaries

| Word | Example | Table | What it does |
|---|---|---|---|
| `average` | `nums.average()` |  | The average (mean) of the numbers; empty values are skipped. |
| `median` | `nums.median()` |  | The middle number once the numbers are sorted; empty values are skipped. |
| `total` | `nums.total()` |  | All the numbers added up; empty values are skipped. |
| `smallest` | `nums.smallest()` |  | The smallest number; empty values are skipped. |
| `biggest` | `nums.biggest()` |  | The biggest number; empty values are skipped. |
| `standard_deviation` | `nums.standard_deviation()` |  | How spread out the numbers are: the sample standard deviation (divide by n - 1). |
| `percentile` | `nums.percentile(90)` |  | The number that this percent of the numbers are below: percentile(50) is the median. |
| `count_each` | `nums.count_each()` |  | Each different number, how many times it appears, and its percent, as a table. |

## Grids of numbers

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.number_grid` | `sdata.number_grid([[1, 2, 3], [4, 5, 6]])` |  | Make a grid of numbers in rows and columns. |
| `number_grid` | `nums.number_grid(columns=3)` |  | Make a NEW grid from these numbers, filled row by row. |
| `count_rows` | `grid.count_rows()` |  | How many rows the grid has. |
| `count_columns` | `grid.count_columns()` |  | How many columns the grid has. |
| `shape` | `grid.shape()` |  | How many rows and columns, as (rows, columns). |
| `show_first_rows` | `grid.show_first_rows(5)` |  | Show the first rows. |
| `show_last_rows` | `grid.show_last_rows(5)` |  | Show the last rows. |
| `number_at` | `grid.number_at(row=1, column=2)` |  | The number at a row and column, counting from 1. |
| `row` | `grid.row(1)` |  | One row of the grid as new numbers, counting from 1: row(1) is the top row. |
| `column` | `grid.column(1)` |  | One column of the grid as new numbers, counting from 1: column(1) is the left one. |
| `numbers` | `grid.numbers()` |  | All the numbers of the grid as a list, row by row (flatten). |
| `swap_rows_and_columns` | `grid.swap_rows_and_columns()` |  | Turn rows into columns and columns into rows (the transpose in maths). |

## Numbers: undo and finish

| Word | Example | Table | What it does |
|---|---|---|---|
| `undo` | `nums.undo()` |  | Take the last step back. |
| `steps` | `nums.steps()` |  | List every step you did, in order. |
| `start_again` | `nums.start_again()` |  | Go back to the numbers as they were at the start. |
| `table` | `nums.table(name="marks")` |  | Turn the numbers into a simplydata table. |
| `chart` | `nums.chart()` |  | Start a chart from these numbers. |
| `save` | `nums.save("nums.csv")` |  | Save the numbers; the end of the file name decides the type. |
| `as_numpy` | `nums.as_numpy()` |  | Give back the numbers as a plain NumPy array, for other Python tools. |

## Help

| Word | Example | Table | What it does |
|---|---|---|---|
| `sdata.words` | `sdata.words()` |  | Print every simplydata word with a short example. |
| `sdata.help` | `sdata.help("bars")` |  | Explain one word: what it does, its options and an example you can run. |
| `sdata.lesson` | `sdata.lesson(1)` |  | A lesson: an example to run, what each line does, and a task. |
