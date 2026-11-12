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

*This file is made by `docs/make_cheatsheet.py`. Do not edit it by hand.*

## Start, show and save

| Word | Example | What it does |
|---|---|---|
| `ps.chart` | `ps.chart("marks.xlsx")` | Start a chart. |
| `ps.example` | `ps.example("marks")` | A small made-up table to practise with. |
| `show` | `chart.show()` | Show the chart in a window. (In Jupyter, VS Code and Spyder charts appear by themselves.) |
| `save` | `chart.save("marks.png")` | Save the chart. The file name decides the format: "marks.png", "marks.pdf". |

## Chart types

| Word | Example | What it does |
|---|---|---|
| `bars` (also `bar`) | `chart.bars(x="name", y="marks")` | Bars. |
| `lines` (also `line`) | `chart.lines(x="day", y="temperature")` | Lines. |
| `dots` (also `scatter`) | `chart.dots(x="height", y="weight")` | Dots (a scatter chart). |
| `area` | `chart.area(x="year", y="sales")` | A line with the space under it coloured in. |
| `histogram` | `chart.histogram(x="marks", bins=10)` | Bars that count how many values fall in each range. |
| `boxes` (also `box`) | `chart.boxes(x="class", y="marks")` | Boxes that show the middle half of the values and the median. |
| `violins` (also `violin`) | `chart.violins(x="class", y="marks")` | Shapes that are wide where many values are. |
| `heatmap` | `chart.heatmap(numbers=True)` | A grid of coloured squares, one colour for each number. |
| `pie` | `chart.pie(labels="fruit", values="count")` | A circle cut into slices, one slice per group. |
| `polar` | `chart.polar(angle="direction", distance="speed")` | Dots on a round chart: how far round, and how far out. |
| `contour` | `chart.contour(x="lon", y="lat", z="depth")` | A map where each colour band joins places with the same height. |
| `surface` | `chart.surface(x="lon", y="lat", z="depth")` | A 3D landscape: z= is how high the ground is. |
| `dots_3d` | `chart.dots_3d(x="a", y="b", z="c")` | Dots floating in a 3D box. |

## Statistics

| Word | Example | What it does |
|---|---|---|
| `show=` | `chart.bars(x="class", y="marks", show="median")` | What each bar or point on a line stands for, when a group has many rows. |
| `error=` | `chart.bars(x="class", y="marks", error="sd")` | Error bars on top of averages: how spread out the values are. |
| `error_bars` | `chart.error_bars("sd")` | Error bars on dots or lines. |
| `fit_line` | `chart.fit_line()` | A straight line that fits the dots best. |
| `fit_curve` | `chart.fit_curve(degree=2)` | A curve that fits the dots best. |
| `smooth_line` | `chart.smooth_line()` | A smooth line that follows the dots (LOWESS). |
| `density` | `chart.density(x="marks")` | A smooth curve that is high where many values are. |
| `count_bars` | `chart.count_bars(x="fruit")` | Bars that count how many rows each group has. |
| `all_points` | `chart.all_points(x="class", y="marks")` | Every value as a dot, spread out so no dot hides another. |
| `side_histograms` | `chart.side_histograms()` | Small histograms along the top and right edges of a dots chart. |
| `ps.pairs` | `ps.pairs("measurements.xlsx")` | Every number column against every other. |

## Lines and shading

| Word | Example | What it does |
|---|---|---|
| `line_at_x` | `chart.line_at_x(10, label="start")` | A straight up-and-down line at one x value. |
| `line_at_y` | `chart.line_at_y(50, label="pass mark")` | A straight side-to-side line at one y value. |
| `shade_between_x` | `chart.shade_between_x(2, 4)` | Shade the band between two x values, behind the data. |
| `shade_between_y` | `chart.shade_between_y(40, 60)` | Shade the band between two y values, behind the data. |

## Words on the chart

| Word | Example | What it does |
|---|---|---|
| `title_top` | `chart.title_top("My class marks")` | A title above the chart. |
| `caption_bottom` | `chart.caption_bottom("Data from 2026")` | A small line of text under the whole chart. |
| `label_x` | `chart.label_x("Student")` | The name under the x-axis. |
| `label_y` | `chart.label_y("Marks")` | The name beside the y-axis. |
| `note` | `chart.note("Highest!", x=5, y=98)` | A note on the chart at a place x, y. |
| `numbers_on_bars` | `chart.numbers_on_bars()` | Write the value of each bar on it. |
| `font_size` | `chart.font_size(14)` | Make all the text bigger or smaller. |

## Axes

| Word | Example | What it does |
|---|---|---|
| `range_x` | `chart.range_x(0, 100)` | From where to where the x-axis goes. |
| `range_y` | `chart.range_y(0, 100)` | From where to where the y-axis goes. |
| `log_scale_x` | `chart.log_scale_x()` | A log scale on the x-axis: 1, 10, 100, 1000 are evenly spaced. |
| `log_scale_y` | `chart.log_scale_y()` | A log scale on the y-axis: 1, 10, 100, 1000 are evenly spaced. |
| `ticks_x` | `chart.ticks_x(0, 10, 20, 30)` | Where the marks on the x-axis go. |
| `ticks_y` | `chart.ticks_y(0, 50, 100)` | Where the marks on the y-axis go. |
| `rotate_labels_x` | `chart.rotate_labels_x()` | Turn the names under the x-axis so long ones do not bump into each other. |
| `rotate_labels_y` | `chart.rotate_labels_y(30)` | Turn the names beside the y-axis. |
| `second_y` | `chart.second_y("rain", chart="bars", label="Rain (mm)")` | A second y-axis on the right, for a column measured in other units. |
| `flip_x` | `chart.flip_x()` | Turn the x-axis around, so numbers grow from right to left. |
| `flip_y` | `chart.flip_y()` | Turn the y-axis upside down, for example for depth going down. |

## Colours and look

| Word | Example | What it does |
|---|---|---|
| `colour` (also `color`) | `chart.colour("blue")` | One colour for the whole chart. |
| `colours` (also `colors`) | `chart.colours("blue", "orange", "green")` | Your own colours for the groups, in order. |
| `style` | `chart.style("school")` | The overall look: "paper" (default), "school", "presentation" or "dark". |
| `size` | `chart.size(width=8, height=6)` | The size in inches. |
| `show_grid` | `chart.show_grid()` | Grid lines across the chart in both directions. |
| `hide_grid` | `chart.hide_grid()` | No grid lines. |
| `legend_top` | `chart.legend_top()` | The legend above the chart, in one row. |
| `legend_bottom` | `chart.legend_bottom()` | The legend below the chart, in one row under the x-axis name. |
| `legend_left` | `chart.legend_left()` | The legend to the left of the chart. |
| `legend_right` | `chart.legend_right()` | The legend to the right of the chart, outside the plot. |
| `hide_legend` | `chart.hide_legend()` | No legend. |
| `colour_bar_label` (also `color_bar_label`) | `chart.colour_bar_label("Depth (m)")` | The name beside the colour bar of a heatmap or contour. |

## Several charts

| Word | Example | What it does |
|---|---|---|
| `one_per` | `chart.one_per("site", columns=3)` | One small chart for each group. |
| `ps.side_by_side` | `ps.side_by_side(a, b)` | Charts next to each other. |
| `ps.one_above_other` | `ps.one_above_other(a, b)` | Charts on top of each other. |
| `ps.grid` | `ps.grid(a, b, c, d, columns=2)` | Charts in rows and columns. |
| `zoom_box` | `chart.zoom_box(x_from=2, x_to=4, y_from=10, y_to=20)` | A small box showing part of the chart close up. |
| `animate` | `chart.animate(over="year")` | A moving chart with one picture for each value of a column. |

## Help

| Word | Example | What it does |
|---|---|---|
| `ps.words` | `ps.words()` | Print every plotsimple word with a short example. |
| `ps.help` | `ps.help("bars")` | Explain one word: what it does, its options and an example you can run. |
| `ps.lesson` | `ps.lesson(1)` | A lesson: an example to run, what each line does, and a task. |
