# plotsimple cheat sheet

Every word on one page. Most words are steps on a chart:

```python
import plotsimple as ps
marks = ps.example("marks")        # a made-up table to practise with
chart = ps.chart(marks)            # or ps.chart("my_file.xlsx")
chart.bars(x="name", y="marks")
chart.title_top("My class marks")
chart.save("marks.png")
```

In Python, `ps.help("bars")` explains any word, `ps.words()` prints this list and
`ps.lesson(1)` starts the one-day course.

Every example runs as written on the practice table in the *Table* column:
for example `weather = ps.example("weather")` then `chart = ps.chart(weather)`.
Steps such as `fit_line` or `title_top` go after a chart step such as `chart.dots(...)`,
and `a`, `b` in `ps.side_by_side(a, b)` are charts you made first.

*This file is made by `docs/make_cheatsheet.py`. Do not edit it by hand.*

## Start, show and save

| Word | Example | Table | What it does |
|---|---|---|---|
| `ps.chart` | `ps.chart(ps.example("marks"))` | marks | Start a chart. |
| `ps.example` | `ps.example("marks")` | marks | A small made-up table to practise with. |
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

## Statistics

| Word | Example | Table | What it does |
|---|---|---|---|
| `show=` | `chart.bars(x="class", y="marks", show="median")` | marks | What each bar or point on a line stands for, when a group has many rows. |
| `error=` | `chart.bars(x="class", y="marks", error="sd")` | marks | Error bars on top of averages: how spread out the values are. |
| `error_bars` | `chart.error_bars("sd")` | marks | Error bars on dots or lines. |
| `fit_line` | `chart.fit_line()` | marks | A straight line that fits the dots best. |
| `fit_curve` | `chart.fit_curve(degree=2)` | weather | A curve that fits the dots best. |
| `smooth_line` | `chart.smooth_line()` | weather | A smooth line that follows the dots (LOWESS). |
| `density` | `chart.density(x="marks", colour="class")` | marks | A smooth curve that is high where many values are. |
| `count_bars` | `chart.count_bars(x="fruit", colour="shop")` | fruit | Bars that count how many rows each group has. |
| `all_points` | `chart.all_points(x="class", y="marks")` | marks | Every value as a dot, spread out so no dot hides another. |
| `side_histograms` | `chart.side_histograms()` | marks | Small histograms along the top and right edges of a dots chart. |
| `ps.pairs` | `ps.pairs(ps.example("marks"), colour="class")` | marks | Every number column against every other. |

## Lines and shading

| Word | Example | Table | What it does |
|---|---|---|---|
| `line_at_x` | `chart.line_at_x(2010, label="2010")` | growth | A straight up-and-down line at one x value. |
| `line_at_y` | `chart.line_at_y(50, label="pass mark")` | marks | A straight side-to-side line at one y value. |
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
| `font_size` | `chart.font_size(14)` | marks | Make all the text bigger or smaller. |

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
| `second_y` | `chart.second_y("rain", chart="bars", label="Rain (mm)")` | weather | A second y-axis on the right, for a column measured in other units. |
| `flip_x` | `chart.flip_x()` | marks | Turn the x-axis around, so numbers grow from right to left. |
| `flip_y` | `chart.flip_y()` | island | Turn the y-axis upside down, for example for depth going down. |

## Colours and look

| Word | Example | Table | What it does |
|---|---|---|---|
| `colour` (also `color`) | `chart.colour("blue")` | marks | One colour for the whole chart. |
| `colours` (also `colors`) | `chart.colours("blue", "orange", "green")` | marks | Your own colours for the groups, in order. |
| `style` | `chart.style("school")` | marks | The overall look. |
| `size` | `chart.size(width=8, height=6)` | weather | The size in inches. |
| `show_grid` | `chart.show_grid()` | marks | Grid lines across the chart in both directions. |
| `hide_grid` | `chart.hide_grid()` | marks | No grid lines. |
| `legend_top` | `chart.legend_top()` | weather | The legend above the chart, in one row. |
| `legend_bottom` | `chart.legend_bottom()` | weather | The legend below the chart, in one row under the x-axis name. |
| `legend_left` | `chart.legend_left()` | weather | The legend to the left of the chart. |
| `legend_right` | `chart.legend_right()` | weather | The legend to the right of the chart, outside the plot. |
| `hide_legend` | `chart.hide_legend()` | weather | No legend. |
| `colour_bar_label` (also `color_bar_label`) | `chart.colour_bar_label("Height (m)")` | island | The name beside the colour bar of a heatmap or contour. |

## Several charts

| Word | Example | Table | What it does |
|---|---|---|---|
| `one_per` | `chart.one_per("class", columns=3)` | marks | One small chart for each group. |
| `ps.side_by_side` | `ps.side_by_side(a, b)` | marks | Charts next to each other. |
| `ps.one_above_other` | `ps.one_above_other(a, b)` | weather | Charts on top of each other. |
| `ps.grid` | `ps.grid(a, b, c, d, columns=2)` | marks | Charts in rows and columns. |
| `zoom_box` | `chart.zoom_box(x_from=2, x_to=4, y_from=40, y_to=60)` | marks | A small box showing part of the chart close up. |
| `animate` | `chart.animate(over="year")` | growth | A moving chart with one picture for each value of a column. |

## Help

| Word | Example | Table | What it does |
|---|---|---|---|
| `ps.words` | `ps.words()` |  | Print every plotsimple word with a short example. |
| `ps.help` | `ps.help("bars")` |  | Explain one word: what it does, its options and an example you can run. |
| `ps.lesson` | `ps.lesson(1)` |  | A lesson: an example to run, what each line does, and a task. |
