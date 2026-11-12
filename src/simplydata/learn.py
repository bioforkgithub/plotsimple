"""Learning simplydata: sdata.words(), sdata.help("bars") and sdata.lesson(1) ... sdata.lesson(15).

The word list is built from the real words and their docstrings, so it cannot go out of
date. Only the grouping (GROUPS) and the runnable examples (HELP_EXAMPLES, LESSONS) are
written by hand, and the tests check that every word is in a group, has an example,
and that every example runs.
"""

from __future__ import annotations

import builtins
import inspect
import re
import textwrap

from ._errors import ISSUES_URL, SimplydataError, suggest

WIDTH = 78

# ---------------------------------------------------------------- the grouping
# "sdata.name" is a function you call on sdata; "name=" is an option used inside chart steps;
# a plain name is a step on a chart (chart.name(...)); "table.name" is a step on a table
# (table.name(...)); "nums.name" is a step on numbers and "grid.name" a step on a grid of
# numbers. Other spellings of a word (bar, scatter, color ...) are found by themselves and
# shown next to it. Old names (kept for one release) are never listed here.
GROUPS = [
    ("Start, show and save", ["sdata.chart", "sdata.example", "show", "save"]),
    ("Chart types", ["bars", "lines", "dots", "area", "histogram", "boxes", "violins",
                     "heatmap", "pie", "polar", "contour", "surface", "dots_3d", "band", "empty_cells"]),
    ("Statistics", ["show=", "error=", "error_bars", "fit_line", "fit_curve", "smooth_line",
                    "density", "count_bars", "all_dots", "side_histograms", "sdata.pairs"]),
    ("Lines and shading", ["line_at_x", "line_at_y", "shade_between_x", "shade_between_y"]),
    ("Words on the chart", ["title_top", "caption_bottom", "label_x", "label_y", "note",
                            "numbers_on_bars", "names_on_dots", "font_size"]),
    ("Axes", ["range_x", "range_y", "log_scale_x", "log_scale_y", "ticks_x", "ticks_y",
              "rotate_labels_x", "rotate_labels_y", "second_y", "flip_x", "flip_y"]),
    ("Colours and look", ["colour", "colours", "style", "size", "show_grid", "hide_grid",
                          "legend_top", "legend_bottom", "legend_left", "legend_right",
                          "hide_legend", "show_legend", "label_legend", "label_colour_bar"]),
    ("Order of groups", ["order_x", "order_y", "order_legend"]),
    ("Several charts", ["one_chart_per", "sdata.side_by_side", "sdata.one_above_other", "sdata.grid",
                        "zoom_box", "animate"]),
    # Table words: "table.name" is a step on a table (table.name(...)).
    ("Tables: start and look", ["sdata.table", "table.show_first_rows", "table.show_last_rows",
                                "table.show", "table.column_names", "table.describe_columns", "table.column"]),
    ("Choose columns", ["table.keep_columns", "table.remove_columns", "table.rename_column"]),
    ("Choose rows", ["table.keep_rows", "table.remove_rows", "table.keep_first_rows",
                     "table.keep_last_rows", "table.keep_random_rows", "table.remove_duplicate_rows",
                     "table.keep_biggest_rows", "table.keep_smallest_rows", "table.show_rows_with_outliers",
                     "table.remove_rows_with_outliers"]),
    ("Empty cells", ["table.count_empty_cells", "table.remove_rows_with_empty_cells",
                     "table.fill_empty_cells", "table.add_is_empty_column"]),
    ("Fix columns", ["table.make_numbers", "table.make_text", "table.make_dates",
                     "table.make_lowercase", "table.make_uppercase", "table.remove_extra_spaces",
                     "table.replace_values", "table.round_to", "table.change_column"]),
    ("Sort and new columns", ["table.sort_by", "table.add_column", "table.add_group_column",
                              "table.add_rank_column"]),
    ("Dates and text", ["table.add_date_part_column", "table.add_text_part_column", "table.split_column",
                        "table.replace_text"]),
    ("Row after row", ["table.add_rolling_column", "table.add_previous_value_column", "table.add_change_column"]),
    ("Summaries", ["table.count_rows", "table.count_each", "table.average_of", "table.median_of",
                   "table.total_of", "table.smallest_of", "table.biggest_of",
                   "table.standard_deviation_of", "table.summary_of", "table.percent_of",
                   "table.count_different_values", "table.correlation_between"]),
    ("Statistical tests", ["table.compare_averages", "table.compare_each_pair", "table.compare_before_after",
                           "table.compare_average_to", "table.compare_counts", "table.fit_straight_line",
                           "table.fit_formula", "table.compare_shape_to", "table.add_adjusted_p_value_column"]),
    ("Combine and reshape", ["table.join_with", "table.add_rows_from", "table.turn_columns_into_rows",
                             "table.turn_rows_into_columns"]),
    ("Undo and finish", ["table.undo", "table.steps", "table.start_again", "table.save",
                         "table.chart", "table.as_pandas", "table.as_polars"]),
    # Number words: "nums.name" is a step on numbers, "grid.name" a step on a grid of numbers.
    ("Numbers: make and look", ["sdata.numbers", "sdata.random_numbers", "nums.show", "nums.show_first_numbers",
                                "nums.show_last_numbers", "nums.number_at", "nums.count",
                                "nums.count_empty_numbers"]),
    ("Numbers: maths in words", ["nums.add", "nums.subtract", "nums.times", "nums.divide_by", "nums.power",
                                 "nums.square_root", "nums.absolute_value", "nums.round_to", "nums.log",
                                 "nums.exp", "nums.running_total"]),
    ("Numbers: choose, sort and fill", ["nums.keep_numbers", "nums.remove_numbers", "nums.keep_first_numbers",
                                        "nums.keep_last_numbers", "nums.keep_random_numbers",
                                        "nums.remove_duplicate_numbers", "nums.remove_empty_numbers",
                                        "nums.fill_empty_numbers", "nums.sort", "nums.reverse"]),
    ("Numbers: summaries", ["nums.average", "nums.median", "nums.total", "nums.smallest", "nums.biggest",
                            "nums.standard_deviation", "nums.percentile", "nums.count_each"]),
    ("Grids of numbers", ["sdata.number_grid", "nums.number_grid", "grid.count_rows", "grid.count_columns",
                          "grid.shape", "grid.show_first_rows", "grid.show_last_rows", "grid.number_at",
                          "grid.row", "grid.column", "grid.numbers", "grid.swap_rows_and_columns"]),
    ("Numbers: undo and finish", ["nums.undo", "nums.steps", "nums.start_again", "nums.table", "nums.chart",
                                  "nums.save", "nums.as_numpy"]),
    ("Help", ["sdata.words", "sdata.help", "sdata.lesson"]),
]

# Options that work inside several chart steps. They are not steps, so they are
# described here.
OPTIONS = {
    "show=": dict(
        about="What each bar or point on a line stands for, when a group has many rows.",
        text=(
            'show="average" (the normal choice) adds up the values and divides by how many '
            'there are. show="median" is the middle value when they are put in order. '
            'show="total" adds them all up. show="count" counts the rows (bars only). '
            'show="smallest", "biggest" and "standard deviation" work too. The names are '
            'the same as the table words: average_of, median_of, total_of, smallest_of ...'
        ),
        words=["bars", "lines"],
        example='chart.bars(x="class", y="marks", show="median")',
    ),
    "error=": dict(
        about="Error bars on top of averages: how spread out, or how sure, the values are.",
        text=(
            'error="standard deviation" shows how far the values usually are from the '
            'average: how spread out they are. error="standard error" shows how sure we are '
            'about the average itself; it gets smaller as you collect more values. '
            'error="confidence interval" is the 95 % confidence interval: the range the true '
            'average is very likely to be in. Lines get a shaded band instead of bars. A line '
            'is printed saying which one was drawn, so your figure caption can say it too. '
            'The short forms "sd", "se" and "ci" also work.'
        ),
        words=["bars", "lines"],
        example='chart.bars(x="class", y="marks", error="standard deviation")',
    ),
}

# A few words have something extra worth saying in their help.
EXTRA_NOTES = {
    "colour": (
        'Inside a chart step, colour= does more: colour="class" gives each class its own '
        'colour (and a legend), and colour="red" makes that one step red.'
    ),
    "show": (
        'Not the same as the show= option inside bars and lines: for that, type '
        'sdata.help("show=").'
    ),
    "error_bars": (
        'Standard deviation or standard error? The standard deviation describes the values '
        '(how spread out they are). The standard error describes the average (how sure we '
        'are of it). Say in your caption which one you drew.'
    ),
    "save": (
        'File types: .png (a picture), .pdf and .svg (sharp at any size, good for printing), '
        '.jpg, .tif, and .gif for animations. quality="high" saves a bigger, sharper picture.'
    ),
}


def _ex(table: str, body: str, file: str) -> str:
    """A runnable example: load an example table, start a chart, do the steps, save."""
    lines = ["import simplydata as sdata", f'{table} = sdata.example("{table}")',
             f"chart = sdata.chart({table})"]
    lines += textwrap.dedent(body).strip().splitlines()
    lines.append(f'chart.save("{file}.png")')
    return "\n".join(lines)


def _tx(name: str, body: str) -> str:
    """A runnable table example: start a table from a practice table, then the steps."""
    lines = ["import simplydata as sdata", f'table = sdata.table(sdata.example("{name}"))']
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


def _nx(body: str, values: str = "[12, 15, 9, 20, 14]") -> str:
    """A runnable numbers example: make some numbers, then the steps."""
    lines = ["import simplydata as sdata", f"nums = sdata.numbers({values})"]
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


def _gx(body: str) -> str:
    """A runnable grid example: make a grid of numbers, then the steps."""
    lines = ["import simplydata as sdata", "grid = sdata.number_grid([[3, 5, 2], [8, 6, 4]])"]
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


HELP_EXAMPLES = {
    # start, show, save
    "sdata.chart": _ex("marks", 'chart.dots(x="hours", y="marks")', "my_chart"),
    "sdata.example": textwrap.dedent("""\
        import simplydata as sdata
        sdata.example()
        marks = sdata.example("marks")
        print(marks)"""),
    "show": _ex("marks", 'chart.bars(x="name", y="marks")\nchart.show()', "marks"),
    "save": textwrap.dedent("""\
        import simplydata as sdata
        marks = sdata.example("marks")
        chart = sdata.chart(marks)
        chart.bars(x="name", y="marks")
        chart.save("marks.png")
        chart.save("marks.pdf")
        chart.save("marks_sharp.png", quality="high")"""),
    # chart types
    "bars": _ex("marks", 'chart.bars(x="name", y="marks", colour="class")\nchart.rotate_labels_x()', "bars"),
    "lines": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city", dots=True)', "lines"),
    "dots": _ex("marks", 'chart.dots(x="hours", y="marks", colour="class", size="height")', "dots"),
    "area": _ex("growth", 'chart.area(x="year", y="population", colour="country", stacked=True)', "area"),
    "histogram": _ex("marks", 'chart.histogram(x="marks", bins=10)', "histogram"),
    "boxes": _ex("marks", 'chart.boxes(x="class", y="marks")', "boxes"),
    "violins": _ex("marks", 'chart.violins(x="subject", y="marks")', "violins"),
    "heatmap": _ex("weather", 'chart.heatmap(x="month", y="city", value="rain", numbers=True)', "heatmap"),
    "pie": _ex("fruit", 'chart.pie(labels="fruit", values="count")', "pie"),
    "polar": _ex("weather", 'chart.polar(angle="wind_direction", distance="wind_speed", colour="city")', "polar"),
    "contour": _ex("island", 'chart.contour(x="east", y="north", z="height")', "contour"),
    "surface": _ex("island", 'chart.surface(x="east", y="north", z="height")', "surface"),
    "dots_3d": _ex("marks", 'chart.dots_3d(x="hours", y="height", z="marks", colour="class")', "dots_3d"),
    "band": textwrap.dedent("""\
        import simplydata as sdata
        table = sdata.table(sdata.example("weather"))
        table.keep_rows("city", equal_to="Sunnyvale")
        table.add_column("lowest", "temperature - 4")
        table.add_column("highest", "temperature + 4")
        chart = table.chart()
        chart.band(x="month", between=["lowest", "highest"], label="4 degrees either side")
        chart.lines(x="month", y="temperature")
        chart.save("band.png")"""),
    "empty_cells": _ex("shop", 'chart.empty_cells()\nchart.title_top("Where the empty cells are")', "empty_cells"),
    # statistics
    "show=": _ex("marks", 'chart.bars(x="class", y="marks", show="median")', "median"),
    "error=": _ex("marks", 'chart.bars(x="class", y="marks", error="standard deviation")', "error"),
    "error_bars": _ex("marks", 'chart.lines(x="subject", y="marks", colour="class")\n'
                      'chart.error_bars("standard error")', "error_bars"),
    "fit_line": _ex("marks", 'chart.dots(x="hours", y="marks")\n'
                    'chart.fit_line(confidence_band=True, show_equation=True)', "fit_line"),
    "fit_curve": _ex("weather", 'chart.dots(x="month_number", y="temperature", colour="city")\nchart.fit_curve(degree=2)', "fit_curve"),
    "smooth_line": _ex("weather", 'chart.dots(x="month_number", y="rain", colour="city")\nchart.smooth_line()', "smooth_line"),
    "density": _ex("marks", 'chart.density(x="marks", colour="class")', "density"),
    "count_bars": _ex("fruit", 'chart.count_bars(x="fruit", colour="shop")', "count_bars"),
    "all_dots": _ex("marks", 'chart.boxes(x="class", y="marks")\nchart.all_dots(x="class", y="marks")', "all_dots"),
    "side_histograms": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.side_histograms()', "side_histograms"),
    "sdata.pairs": textwrap.dedent("""\
        import simplydata as sdata
        marks = sdata.example("marks")
        chart = sdata.pairs(marks, colour="class")
        chart.title_top("Every number column against every other")
        chart.save("pairs.png")"""),
    # lines and shading
    "line_at_x": _ex("growth", 'chart.lines(x="year", y="population", colour="country")\nchart.line_at_x(2010, label="2010")', "line_at_x"),
    "line_at_y": _ex("marks", 'chart.bars(x="name", y="marks")\nchart.line_at_y(50, label="pass mark")\nchart.rotate_labels_x()', "line_at_y"),
    "shade_between_x": _ex("weather", 'chart.lines(x="month_number", y="temperature", colour="city")\nchart.shade_between_x(6, 8)', "shade_between_x"),
    "shade_between_y": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.shade_between_y(40, 60)', "shade_between_y"),
    # words on the chart
    "title_top": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.title_top("Marks in each class")', "title_top"),
    "caption_bottom": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.caption_bottom("Made-up data from sdata.example")', "caption_bottom"),
    "label_x": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.label_x("Hours of study")', "label_x"),
    "label_y": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.label_y("Marks (out of 100)")', "label_y"),
    "note": _ex("growth", 'chart.lines(x="year", y="population", colour="country")\nchart.note("Cherria grows fastest", x=2020, y=12, arrow=True)', "note"),
    "names_on_dots": textwrap.dedent("""\
        import simplydata as sdata
        table = sdata.table(sdata.example("marks"))
        table.keep_rows("subject", equal_to="Maths")
        chart = table.chart()
        chart.dots(x="hours", y="marks")
        chart.names_on_dots("name")
        chart.save("names_on_dots.png")"""),
    "numbers_on_bars": _ex("fruit", 'chart.bars(x="fruit", y="count", show="total")\nchart.numbers_on_bars()', "numbers_on_bars"),
    "font_size": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.title_top("Big text")\nchart.font_size(18)', "font_size"),
    # axes
    "range_x": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.range_x(0, 10)', "range_x"),
    "range_y": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.range_y(0, 100)', "range_y"),
    "log_scale_x": _ex("growth", 'chart.dots(x="population", y="forest", colour="country")\nchart.log_scale_x()', "log_scale_x"),
    "log_scale_y": _ex("growth", 'chart.lines(x="year", y="population", colour="country")\nchart.log_scale_y()', "log_scale_y"),
    "ticks_x": _ex("weather", 'chart.lines(x="month_number", y="rain", colour="city")\n'
                   'chart.ticks_x(1, 4, 7, 10, labels=["Jan", "Apr", "Jul", "Oct"])', "ticks_x"),
    "ticks_y": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.ticks_y(0, 25, 50, 75)', "ticks_y"),
    "rotate_labels_x": _ex("marks", 'chart.bars(x="name", y="marks")\nchart.rotate_labels_x(45)', "rotate_labels_x"),
    "rotate_labels_y": _ex("fruit", 'chart.bars(x="fruit", y="count", sideways=True)\nchart.rotate_labels_y(30)', "rotate_labels_y"),
    "second_y": _ex("weather", 'chart.lines(x="month", y="temperature")\nchart.second_y("rain", drawn_as="bars", label="Rain (mm)")', "second_y"),
    "flip_x": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.flip_x()', "flip_x"),
    "flip_y": _ex("island", 'chart.contour(x="east", y="north", z="height")\nchart.flip_y()', "flip_y"),
    # colours and look
    "colour": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.colour("purple")', "colour"),
    "colours": _ex("marks", 'chart.bars(x="subject", y="marks", colour="class")\nchart.colours("orange", "teal", "purple")', "colours"),
    "style": _ex("marks", 'chart.bars(x="subject", y="marks", colour="class")\nchart.style("school")', "style"),
    "size": _ex("weather", 'chart.lines(x="month", y="rain", colour="city")\nchart.size(width=10, height=4)', "size"),
    "show_grid": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.show_grid()', "show_grid"),
    "hide_grid": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.hide_grid()', "hide_grid"),
    "legend_top": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.legend_top()', "legend_top"),
    "legend_bottom": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.legend_bottom()', "legend_bottom"),
    "legend_left": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.legend_left()', "legend_left"),
    "legend_right": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.legend_right()', "legend_right"),
    "hide_legend": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.hide_legend()', "hide_legend"),
    "show_legend": _ex("weather", 'chart.lines(x="month", y="temperature", colour="city")\nchart.hide_legend()\n'
                       'chart.show_legend()', "show_legend"),
    "label_legend": _ex("marks", 'chart.dots(x="hours", y="marks", colour="class")\nchart.label_legend("Class")',
                        "label_legend"),
    "order_x": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.order_x("C", "B", "A")', "order_x"),
    "order_y": _ex("marks", 'chart.bars(x="class", y="marks", sideways=True)\nchart.order_y("C", "B", "A")', "order_y"),
    "order_legend": _ex("marks", 'chart.bars(x="class", y="marks", colour="subject")\n'
                        'chart.order_legend("Science", "Maths", "English")', "order_legend"),
    "label_colour_bar": _ex("island", 'chart.contour(x="east", y="north", z="height")\n'
                            'chart.label_colour_bar("Height (m)")', "label_colour_bar"),
    # several charts
    "one_chart_per": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.one_chart_per("class", per_row=3)',
                         "one_chart_per"),
    "sdata.side_by_side": textwrap.dedent("""\
        import simplydata as sdata
        marks = sdata.example("marks")
        a = sdata.chart(marks).bars(x="class", y="marks").title_top("Average")
        b = sdata.chart(marks).boxes(x="class", y="marks").title_top("Spread")
        page = sdata.side_by_side(a, b, same_y=True)
        page.save("side_by_side.png")"""),
    "sdata.one_above_other": textwrap.dedent("""\
        import simplydata as sdata
        weather = sdata.example("weather")
        a = sdata.chart(weather).lines(x="month", y="temperature", colour="city")
        b = sdata.chart(weather).bars(x="month", y="rain", colour="city")
        page = sdata.one_above_other(a, b)
        page.save("one_above_other.png")"""),
    "sdata.grid": textwrap.dedent("""\
        import simplydata as sdata
        marks = sdata.example("marks")
        a = sdata.chart(marks).bars(x="class", y="marks")
        b = sdata.chart(marks).histogram(x="marks")
        c = sdata.chart(marks).dots(x="hours", y="marks")
        d = sdata.chart(marks).boxes(x="subject", y="marks")
        page = sdata.grid(a, b, c, d, per_row=2)
        page.title_top("Four ways to look at the marks")
        page.save("grid.png")"""),
    "zoom_box": _ex("marks", 'chart.dots(x="hours", y="marks")\n'
                    'chart.zoom_box(x_between=[2, 4], y_between=[40, 60], where="top left")', "zoom_box"),
    "animate": textwrap.dedent("""\
        import simplydata as sdata
        growth = sdata.example("growth")
        chart = sdata.chart(growth)
        chart.bars(x="country", y="population")
        chart.range_y(0, 50)
        chart.animate(over="year")
        chart.save("growth.gif")"""),
    # tables: start and look
    "sdata.table": _tx("marks", "table.show_first_rows()"),
    "table.show_first_rows": _tx("marks", "table.show_first_rows(10)"),
    "table.show_last_rows": _tx("marks", "table.show_last_rows(3)"),
    "table.show": _tx("marks", 'table.keep_rows("class", equal_to="A")\ntable.show(rows=10)'),
    "table.column_names": _tx("marks", "print(table.column_names())"),
    "table.column": _tx("marks", 'marks = table.column("marks")\nmarks.average()\nmarks.percentile(90)'),
    "table.describe_columns": _tx("marks", "table.describe_columns()"),
    # choose columns
    "table.keep_columns": _tx("marks", 'table.keep_columns("name", "marks")\ntable.show_first_rows()'),
    "table.remove_columns": _tx("marks", 'table.remove_columns("height")\nprint(table.column_names())'),
    "table.rename_column": _tx("marks", 'table.rename_column("marks", to="score")\ntable.show_first_rows()'),
    # choose rows
    "table.keep_rows": _tx("marks", 'table.keep_rows("marks", above=50)\n'
                                    'table.keep_rows("subject", one_of=["Maths", "Science"])\n'
                                    "table.keep_rows(\"hours > 5 and class != 'C'\")\ntable.show_first_rows()"),
    "table.remove_rows": _tx("marks", 'table.remove_rows("subject", equal_to="English")\ntable.show_first_rows()'),
    "table.keep_first_rows": _tx("marks", "table.keep_first_rows(10)\nprint(table)"),
    "table.keep_last_rows": _tx("marks", "table.keep_last_rows(10)\nprint(table)"),
    "table.keep_random_rows": _tx("marks", "table.keep_random_rows(10, seed=1)\nprint(table)"),
    "table.remove_duplicate_rows": _tx("marks", 'table.remove_duplicate_rows(columns=["name"])\nprint(table)'),
    "table.keep_biggest_rows": _tx("marks", 'table.keep_biggest_rows(3, by="marks", for_each="subject")\ntable.show()'),
    "table.keep_smallest_rows": _tx("marks", 'table.keep_smallest_rows(3, by="marks", for_each="subject")\ntable.show()'),
    "table.show_rows_with_outliers": _tx("marks", 'table.show_rows_with_outliers("marks", for_each="class")'),
    "table.remove_rows_with_outliers": _tx("marks", 'table.remove_rows_with_outliers("marks", for_each="subject")\n'
                                                    "table.steps()"),
    # empty cells (the shop table has some)
    "table.count_empty_cells": _tx("shop", "table.count_empty_cells()"),
    "table.remove_rows_with_empty_cells": _tx("shop", 'table.remove_rows_with_empty_cells("sales")\nprint(table)'),
    "table.fill_empty_cells": _tx("shop", 'table.fill_empty_cells("customers", with_the="median")\n'
                                          'table.fill_empty_cells("sales", with_the="line between neighbours", '
                                          'sorted_by="date", for_each="shop")\ntable.count_empty_cells()'),
    "table.add_is_empty_column": _tx("shop", 'table.add_is_empty_column("sales_was_empty", from_column="sales")\n'
                                             'table.fill_empty_cells("sales", with_the="line between neighbours", '
                                             'sorted_by="date", for_each="shop")\n'
                                             'table.keep_rows("sales_was_empty")\ntable.show()'),
    # fix columns
    "table.make_numbers": _tx("marks", 'table.make_numbers("marks")\ntable.describe_columns()'),
    "table.make_text": _tx("weather", 'table.make_text("month_number")\ntable.describe_columns()'),
    "table.make_dates": _tx("growth", 'table.make_dates("year")\ntable.show_first_rows()'),
    "table.make_lowercase": _tx("marks", 'table.make_lowercase("name")\ntable.show_first_rows()'),
    "table.make_uppercase": _tx("marks", 'table.make_uppercase("name")\ntable.show_first_rows()'),
    "table.remove_extra_spaces": _tx("marks", 'table.remove_extra_spaces("name")\ntable.show_first_rows()'),
    "table.replace_values": _tx("marks", 'table.replace_values("subject", "Maths", with_value="Math")\n'
                                         'table.count_each("subject")'),
    "table.round_to": _tx("marks", 'table.round_to("hours", 0)\ntable.show_first_rows()'),
    "table.change_column": _tx("marks", 'table.change_column("marks", "marks / 10")\ntable.show_first_rows()'),
    # sort and new columns
    "table.sort_by": _tx("marks", 'table.sort_by("marks", biggest_first=True)\ntable.show_first_rows()'),
    "table.add_column": _tx("marks", 'table.add_column("study_and_marks", by_adding=["hours", "marks"])\n'
                                     'table.add_column("marks_per_hour", "marks / hours")\ntable.show_first_rows()'),
    "table.add_group_column": _tx("marks", 'table.add_group_column("grade", from_column="marks", '
                                           'cut_at=[40, 60, 80], names=["D", "C", "B", "A"])\n'
                                           'table.count_each("grade")'),
    "table.add_rank_column": _tx("marks", 'table.add_rank_column("place", from_column="marks")\n'
                                          'table.sort_by("place")\ntable.show_first_rows()'),
    # dates and text
    "table.add_date_part_column": _tx("shop", 'table.add_date_part_column("weekday", from_column="date", '
                                              'part="weekday name")\n'
                                              'table.average_of("sales", for_each="weekday")'),
    "table.add_text_part_column": _tx("marks", 'table.add_text_part_column("initial", from_column="name", '
                                               'first_characters=1)\ntable.show_first_rows()'),
    "table.split_column": _tx("fruit", 'table.split_column("shop", separator=" ", into=["shop_kind", "shop_word"])\n'
                                       "table.show_first_rows()"),
    "table.replace_text": _tx("fruit", 'table.replace_text("shop", " shop", with_value="")\ntable.count_each("shop")'),
    # row after row
    "table.add_rolling_column": _tx("weather", 'table.add_rolling_column("rain_3_months", from_column="rain", window=3, '
                                               'show="total", for_each="city", sorted_by="month_number")\n'
                                               "table.show_first_rows()"),
    "table.add_previous_value_column": _tx("weather", 'table.add_previous_value_column("last_month", '
                                                      'from_column="temperature", for_each="city", '
                                                      'sorted_by="month_number")\ntable.show_first_rows()'),
    "table.add_change_column": _tx("growth", 'table.add_change_column("growth", from_column="population", '
                                             'in_percent=True, for_each="country", sorted_by="year")\n'
                                             "table.show_first_rows()"),
    # summaries
    "table.count_rows": _tx("marks", 'table.count_rows()\ntable.count_rows(for_each="class")'),
    "table.count_each": _tx("fruit", 'table.count_each("fruit")'),
    "table.average_of": _tx("marks", 'table.average_of("marks")\n'
                                     'by_class = table.average_of("marks", for_each="class")\nprint(by_class)'),
    "table.median_of": _tx("marks", 'table.median_of("marks", for_each="class")'),
    "table.total_of": _tx("fruit", 'table.total_of("count", for_each="shop")'),
    "table.smallest_of": _tx("weather", 'table.smallest_of("temperature", for_each="city")'),
    "table.biggest_of": _tx("weather", 'table.biggest_of("temperature", for_each="city")'),
    "table.standard_deviation_of": _tx("marks", 'table.standard_deviation_of("marks", for_each="class")'),
    "table.summary_of": _tx("marks", 'table.summary_of("marks", for_each="class", '
                                     'show=["average", "standard deviation", "count"])'),
    "table.percent_of": _tx("marks", 'table.percent_of("marks", above=60, for_each="class")'),
    "table.count_different_values": _tx("marks", 'table.count_different_values("name", for_each="class")'),
    "table.correlation_between": _tx("marks", 'table.correlation_between("hours", "marks")'),
    # statistical tests
    "table.compare_averages": _tx("marks", 'result = table.compare_averages("marks", for_each="class")\n'
                                           'table.compare_averages("marks", for_each="class", based_on="ranks")\n'
                                           'result.save("marks_test.csv")'),
    "table.compare_each_pair": _tx("marks", 'table.compare_each_pair("marks", for_each="class")'),
    "table.compare_before_after": _tx("marks", 'table.keep_columns("name", "subject", "marks")\n'
                                               'table.turn_rows_into_columns(names_from="subject", values_from="marks")\n'
                                               'table.compare_before_after("Maths", "Science")'),
    "table.compare_average_to": _tx("marks", 'table.compare_average_to("marks", 60)'),
    "table.compare_counts": _tx("marks", 'table.add_group_column("result", from_column="marks", cut_at=[60], '
                                         'names=["below 60", "60 or more"])\n'
                                         'table.compare_counts("result", for_each="class")'),
    "table.fit_straight_line": _tx("marks", 'table.fit_straight_line(x="hours", y="marks")'),
    "table.fit_formula": _tx("growth", 'table.keep_rows("country", equal_to="Applia")\n'
                                       'table.fit_formula("population = a * exp(b * (year - 2000))")'),
    "table.compare_shape_to": _tx("marks", 'table.compare_shape_to("marks", "normal", for_each="class")'),
    "table.add_adjusted_p_value_column": _tx("marks", 'table = table.correlation_between("hours", "marks", "height", '
                                                      'show="each pair")\n'
                                                      'table.add_adjusted_p_value_column("q", method="benjamini-hochberg")\n'
                                                      "table.show()"),
    # combine and reshape
    "table.join_with": _tx("marks", 'ages = {"name": ["Asha", "Ben"], "age": [12, 13]}\n'
                                    'table.join_with(ages, using="name", keep="all rows from this table")\n'
                                    "table.show_first_rows()"),
    "table.add_rows_from": _tx("marks", 'table.add_rows_from(sdata.example("marks"))\nprint(table)'),
    "table.turn_columns_into_rows": _tx("weather", 'table.keep_columns("month", "city", "temperature", "rain")\n'
                                                   'table.turn_columns_into_rows(["temperature", "rain"])\n'
                                                   "table.show_first_rows()"),
    "table.turn_rows_into_columns": _tx("weather", 'table.turn_rows_into_columns(names_from="city", '
                                                   'values_from="temperature")\ntable.show_first_rows()'),
    # undo and finish
    "table.undo": _tx("marks", 'table.keep_rows("marks", above=50)\ntable.undo()\nprint(table)'),
    "table.steps": _tx("marks", 'table.keep_rows("marks", above=50)\ntable.sort_by("marks")\ntable.steps()'),
    "table.start_again": _tx("marks", 'table.keep_rows("marks", above=50)\ntable.remove_columns("height")\n'
                                      "table.start_again()\nprint(table)"),
    "table.save": _tx("marks", 'table.keep_rows("marks", above=50)\ntable.save("good_marks.xlsx")\n'
                               'table.save("good_marks.csv")'),
    "table.chart": _tx("marks", 'table.keep_rows("subject", equal_to="Maths")\nchart = table.chart()\n'
                                'chart.bars(x="name", y="marks")\nchart.rotate_labels_x()\n'
                                'chart.save("maths_marks.png")'),
    "table.as_pandas": _tx("marks", 'table.keep_rows("marks", above=50)\ndf = table.as_pandas()\nprint(type(df))'),
    "table.as_polars": _tx("marks", 'table.keep_rows("marks", above=50)\ndf = table.as_polars()\nprint(type(df))'),
    # numbers: make and look
    "sdata.numbers": textwrap.dedent("""\
        import simplydata as sdata
        nums = sdata.numbers([3, 5, 2, 8])
        print(nums)
        print(sdata.numbers(1, to=10))
        print(sdata.numbers(0, to=1, how_many=5))"""),
    "sdata.random_numbers": textwrap.dedent("""\
        import simplydata as sdata
        dice = sdata.random_numbers(how_many=10, between=[1, 6], whole_numbers=True, seed=1)
        dice.show()
        heights = sdata.random_numbers(how_many=100, average=150, standard_deviation=8, seed=1)
        heights.average()"""),
    "nums.show": _nx("nums.show()"),
    "nums.show_first_numbers": _nx("nums.show_first_numbers(3)"),
    "nums.show_last_numbers": _nx("nums.show_last_numbers(2)"),
    "nums.number_at": _nx("print(nums.number_at(1))\nprint(nums.number_at(nums.count()))"),
    "nums.count": _nx("print(nums.count())"),
    "nums.count_empty_numbers": _nx("print(nums.count_empty_numbers())", "[12, None, 9, None, 14]"),
    # numbers: maths in words
    "nums.add": _nx("nums.add(5)\nnums.show()"),
    "nums.subtract": _nx("nums.subtract(2)\nnums.show()"),
    "nums.times": _nx("nums.times(2)\nnums.show()"),
    "nums.divide_by": _nx("nums.divide_by(4)\nnums.show()"),
    "nums.power": _nx("nums.power(2)\nnums.show()"),
    "nums.square_root": _nx("nums.square_root()\nnums.show()", "[9, 16, 25]"),
    "nums.absolute_value": _nx("nums.absolute_value()\nnums.show()", "[-3, 5, -2]"),
    "nums.round_to": _nx("nums.round_to(1)\nnums.show()", "[2.345, 3.14159, 7.07]"),
    "nums.log": _nx("nums.log(base=10)\nnums.show()", "[1, 10, 100, 1000]"),
    "nums.exp": _nx("nums.exp()\nnums.show()", "[0, 1, 2]"),
    "nums.running_total": _nx("nums.running_total()\nnums.show()", "[1, 2, 3, 4]"),
    # numbers: choose, sort and fill
    "nums.keep_numbers": _nx("nums.keep_numbers(above=10)\nnums.show()"),
    "nums.remove_numbers": _nx("nums.remove_numbers(below=10)\nnums.show()"),
    "nums.keep_first_numbers": _nx("nums.keep_first_numbers(3)\nnums.show()"),
    "nums.keep_last_numbers": _nx("nums.keep_last_numbers(3)\nnums.show()"),
    "nums.keep_random_numbers": _nx("nums.keep_random_numbers(3, seed=1)\nnums.show()"),
    "nums.remove_duplicate_numbers": _nx("nums.remove_duplicate_numbers()\nnums.show()", "[3, 5, 3, 8, 5]"),
    "nums.remove_empty_numbers": _nx("nums.remove_empty_numbers()\nnums.show()", "[12, None, 9, 14]"),
    "nums.fill_empty_numbers": _nx('nums.fill_empty_numbers(with_the="line between neighbours")\nnums.show()',
                                   "[12, None, 9, None, 14]"),
    "nums.sort": _nx("nums.sort(biggest_first=True)\nnums.show()"),
    "nums.reverse": _nx("nums.reverse()\nnums.show()"),
    # numbers: summaries
    "nums.average": _nx("nums.average()"),
    "nums.median": _nx("nums.median()"),
    "nums.total": _nx("nums.total()"),
    "nums.smallest": _nx("nums.smallest()"),
    "nums.biggest": _nx("nums.biggest()"),
    "nums.standard_deviation": _nx("nums.standard_deviation()"),
    "nums.percentile": _nx("nums.percentile(90)"),
    "nums.count_each": _nx("nums.count_each()", "[3, 5, 3, 8, 5, 3]"),
    # grids of numbers
    "sdata.number_grid": textwrap.dedent("""\
        import simplydata as sdata
        grid = sdata.number_grid([[1, 2, 3], [4, 5, 6]])
        print(grid)
        print(sdata.number_grid(rows=2, columns=3, filled_with=0))"""),
    "nums.number_grid": _nx("grid = nums.number_grid(columns=3)\nprint(grid)", "[1, 2, 3, 4, 5, 6]"),
    "grid.count_rows": _gx("print(grid.count_rows())"),
    "grid.count_columns": _gx("print(grid.count_columns())"),
    "grid.shape": _gx("print(grid.shape())"),
    "grid.show_first_rows": _gx("grid.show_first_rows(1)"),
    "grid.show_last_rows": _gx("grid.show_last_rows(1)"),
    "grid.number_at": _gx("print(grid.number_at(row=1, column=2))"),
    "grid.row": _gx("top = grid.row(1)\ntop.show()"),
    "grid.column": _gx("left = grid.column(1)\nleft.show()"),
    "grid.numbers": _gx("nums = grid.numbers()\nnums.show()"),
    "grid.swap_rows_and_columns": _gx("grid.swap_rows_and_columns()\nprint(grid)"),
    # numbers: undo and finish
    "nums.undo": _nx("nums.add(5)\nnums.undo()\nnums.show()"),
    "nums.steps": _nx("nums.add(5)\nnums.keep_numbers(above=15)\nnums.steps()"),
    "nums.start_again": _nx("nums.add(5)\nnums.times(2)\nnums.start_again()\nnums.show()"),
    "nums.table": _nx('table = nums.table(name="marks")\nprint(table)'),
    "nums.chart": _nx('chart = nums.chart()\nchart.histogram(x="value", bins=4)\nchart.save("nums.png")'),
    "nums.save": _nx('nums.save("nums.csv")'),
    "nums.as_numpy": _nx("array = nums.as_numpy()\nprint(type(array))"),
    # help
    "sdata.words": "import simplydata as sdata\nsdata.words()",
    "sdata.help": 'import simplydata as sdata\nsdata.help("bars")\nsdata.help("show=")\nsdata.help("keep_rows")\nsdata.help("average")',
    "sdata.lesson": "import simplydata as sdata\nsdata.lesson()\nsdata.lesson(1)",
}


# ---------------------------------------------------------------- finding words
def _ps():
    import simplydata  # late import: simplydata imports this file

    return simplydata


def _chart_class():
    from .chart import Chart

    return Chart


def _table_class():
    from .table import Table

    return Table


def _numbers_class():
    from .numbers.numbers import Numbers

    return Numbers


def _grid_class():
    from .numbers.numbers import NumberGrid

    return NumberGrid


# The objects that have steps, and the name they go by in examples ("chart" has no prefix
# in GROUPS: a plain name is a chart word).
_PREFIX = {"table": "table", "numbers": "nums", "grid": "grid"}


def _classes():
    """(class, kind) for every object with steps."""
    return [(_chart_class(), "chart"), (_table_class(), "table"), (_numbers_class(), "numbers"),
            (_grid_class(), "grid")]


def _kind(key: str) -> str:
    """What a key in GROUPS is: "option", "sdata", "table", "numbers", "grid" or "chart"."""
    if key.endswith("="):
        return "option"
    if key.startswith("sdata."):
        return "sdata"
    for kind, prefix in _PREFIX.items():
        if key.startswith(prefix + "."):
            return kind
    return "chart"


def _word(key: str) -> str:
    """The word itself, without "sdata.", "table." or "nums." in front: "sdata.chart" -> "chart"."""
    kind = _kind(key)
    if kind == "sdata":
        return key[len("sdata."):]
    if kind in _PREFIX:
        return key[len(_PREFIX[kind]) + 1:]
    return key


def _display(key: str) -> str:
    """How a key is shown in sdata.words() and the cheat sheet: table and number words without
    "table." or "nums." (their group says what they belong to)."""
    return _word(key) if _kind(key) in _PREFIX else key


def _function(key: str):
    """The real function behind a word in GROUPS (None for options such as "show=")."""
    kind = _kind(key)
    if kind == "option":
        return None
    if kind == "sdata":
        return getattr(_ps(), _word(key))
    for cls, cls_kind in _classes():
        if cls_kind == kind:
            return getattr(cls, _word(key))
    return None  # pragma: no cover


def _all_keys() -> list[str]:
    return [k for _, keys in GROUPS for k in keys]


def _group_of(key: str) -> str:
    return next(g for g, keys in GROUPS if key in keys)


def _aliases() -> dict[str, list[str]]:
    """Other names for the same step: {"bars": ["bar"], "dots": ["scatter"], ...}"""
    found: dict[str, list[str]] = {}
    for cls, kind in _classes():
        keys = [k for k in _all_keys() if _kind(k) == kind]
        words = {_word(k) for k in keys}
        for name in cls._word_names():
            if name in words:
                continue
            func = getattr(cls, name)
            for key in keys:
                if getattr(cls, _word(key)) is func:
                    found.setdefault(key, []).append(name)
    return found


def _call_form(key: str) -> str:
    kind = _kind(key)
    if kind == "option":
        return key
    if kind == "chart":
        return f"chart.{key}()"
    return f"{key}()"


def _prefixes(key: str) -> str:
    """What may come before the word in an example: table.keep_rows(...), chart.bars(...)."""
    return _PREFIX.get(_kind(key), "chart|sdata|page")


def _find_call(text: str, prefix_word: str, prefixes: str = "chart|sdata|page|table|nums|grid") -> str | None:
    """Find the first call like chart.bars(...) in text, brackets and quotes included."""
    m = re.search(rf"(?:{prefixes})\.{re.escape(prefix_word)}\(", text)
    if not m:
        return None
    depth, quote = 0, None
    for i in range(m.end() - 1, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[m.start(): i + 1]
    return None


def _short_example(key: str) -> str:
    """One short example for the word list: from the docstring, else from the help example."""
    if key in OPTIONS:
        return OPTIONS[key]["example"]
    word, prefixes = _word(key), _prefixes(key)
    doc = inspect.getdoc(_function(key)) or ""
    call = _find_call(doc, word, prefixes) or _find_call(HELP_EXAMPLES.get(key, ""), word, prefixes)
    return call or _call_form(key)


_TABLE = re.compile(r'sdata\.example\("(\w+)"\)')


def _example_table(key: str) -> str | None:
    """The example table a word's examples use: named in the docstring, else in the help example.

    A docstring names its table like this: chart.lines(...) (with sdata.example("weather")).
    """
    doc = "" if key in OPTIONS else inspect.getdoc(_function(key)) or ""
    found = _TABLE.search(doc) or _TABLE.search(HELP_EXAMPLES.get(key, ""))
    return found.group(1) if found else None


def _summary(key: str) -> str:
    """The first sentence of the word's docstring (without its example)."""
    if key in OPTIONS:
        return OPTIONS[key]["about"]
    doc = inspect.getdoc(_function(key)) or ""
    first = doc.split("\n\n")[0].replace("\n", " ")
    call = _find_call(first, _word(key), _prefixes(key))
    if call:
        first = first.split(call)[0]
        first = re.sub(r"\s*\w+\s*=\s*$", "", first)  # "... chart = table.chart()": drop "chart ="
    first = first.strip().rstrip(":").strip()
    if first and first[-1] not in ".!?)":
        first += "."
    return first


# ---------------------------------------------------------------- sdata.words()
def words() -> None:
    """Print every simplydata word with a short example. sdata.words()

    For more about one word, type sdata.help("bars").
    """
    aliases = _aliases()
    pad = max(len(_display(k)) for k in _all_keys()) + 2
    print("simplydata words. Type sdata.help(\"bars\") to learn more about any of them.")
    print("Chart words are steps on a chart: chart = sdata.chart(table), then chart.bars(...).")
    print("Table words are steps on a table: table = sdata.table(...), then table.keep_rows(...).")
    print("Number words are steps on numbers: nums = sdata.numbers([3, 5, 2]), then nums.average();")
    print("grid words are steps on a grid of numbers: grid = sdata.number_grid([[1, 2], [3, 4]]).")
    print('The examples use the practice tables; sdata.help("word") shows which one, ready to run.')
    for group, keys in GROUPS:
        print()
        print(group)
        print("-" * len(group))
        for key in keys:
            line = f"  {_display(key):<{pad}}{_short_example(key)}"
            if key in aliases:
                line += "   (also: " + ", ".join(aliases[key]) + ")"
            print(line)
    print()
    print(f"Missing a word? Ask for it at {ISSUES_URL}")


# ---------------------------------------------------------------- sdata.help()
def _normalise(word: str) -> str:
    w = str(word).strip()
    for start in ("sdata.", "chart.", "table.", "page.", "nums.", "grid.", "."):
        if w.startswith(start):
            w = w[len(start):]
    if w.endswith("()"):
        w = w[:-2]
    return w.strip()


def _chart_or_ps_key(w: str) -> str | None:
    """A chart word, sdata word or option called w (already normalised), or None."""
    keys = _all_keys()
    if w in keys:
        return w
    if f"sdata.{w}" in keys:
        return f"sdata.{w}"
    if f"{w}=" in keys:
        return f"{w}="
    if w.endswith("=") and w[:-1] in keys:
        return w[:-1]
    for key, names in _aliases().items():
        if _kind(key) in ("chart", "sdata") and w in names:
            return key
    if w == "Chart":
        return "sdata.chart"
    if w == "Table":
        return "sdata.table"
    return None


def _object_key(w: str, kind: str = "table") -> str | None:
    """A table, numbers or grid word called w (already normalised), or None."""
    if f"{_PREFIX[kind]}.{w}" in _all_keys():
        return f"{_PREFIX[kind]}.{w}"
    for key, names in _aliases().items():
        if _kind(key) == kind and w in names:
            return key
    return None


def _table_key(w: str) -> str | None:
    """A table word called w (already normalised), or None."""
    return _object_key(w, "table")


def _lookup_all(word) -> list[str]:
    """Every key of GROUPS that what the user typed can mean.

    "save" is a word for charts, tables and numbers, so it gives all of them; "chart.save"
    and "table.save" give one each.
    """
    typed = str(word).strip()
    w = _normalise(typed)
    for kind, prefix in _PREFIX.items():
        if typed.startswith(prefix + "."):
            found = [_object_key(w, kind)]
            break
    else:
        if typed.startswith(("chart.", "page.", "sdata.")):
            found = [_chart_or_ps_key(w)]
        else:
            found = [_chart_or_ps_key(w)] + [_object_key(w, kind) for kind in _PREFIX]
    return [k for k in dict.fromkeys(found) if k is not None]


def _old_names(word) -> list[tuple[str, str, str]]:
    """What an old name is called now: [(old, new, key of the new word), ...].

    Old words come from each object's _OLD_WORDS, old options from the renamed= of
    @friendly. Both still work for one release, but sdata.words() never lists them.
    """
    w = _normalise(str(word))
    found = []
    for cls, kind in _classes():
        new = cls._OLD_WORDS.get(w)
        if new is None:
            continue
        new = new[0] if isinstance(new, tuple) else new
        key = new if kind == "chart" else f"{_PREFIX[kind]}.{new}"
        if key in _all_keys():
            found.append((w, new, key))
    option = w[:-1] if w.endswith("=") else w
    for key in _all_keys():
        func = _function(key)
        new = getattr(func, "_renamed_options", {}).get(option)
        if new:
            found.append((f"{option}=", f"{new}=", key))
    return list(dict.fromkeys(found))


def _lookup(word) -> str | None:
    """Turn what the user typed into a key of GROUPS (the first, if it has two), or None."""
    found = _lookup_all(word)
    return found[0] if found else None


def _options_text(key: str) -> list[str]:
    func = _function(key)
    if func is None:
        return []
    params = [p for p in inspect.signature(func).parameters.values()
              if p.name != "self" and not (p.kind is p.VAR_POSITIONAL and p.name == "extra")]
    if not params:
        return ["  (nothing: just write the empty brackets)"]
    width = max(len(p.name) for p in params) + 4
    lines = []
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            lines.append(f"  {p.name + '...':<{width}}as many as you like, separated by commas")
        elif p.kind is p.VAR_KEYWORD:
            continue
        elif p.default is p.empty:
            lines.append(f"  {p.name:<{width}}needed")
        elif p.default is None or type(p.default).__name__ == "_NotGiven":
            lines.append(f"  {p.name}=")
        else:
            normal = f'"{p.default}"' if isinstance(p.default, str) else repr(p.default)
            lines.append(f"  {p.name + '=':<{width}}normally {normal}")
    return lines


def _print_help_overview() -> None:
    print(textwrap.dedent("""\
        How to get help in simplydata

          sdata.help("bars")       explains one word: what it does, its options and an example
          sdata.help("keep_rows")  works for table words too, and sdata.help("average") for numbers
          sdata.help("show=")      explains an option you use inside a step
          sdata.words()            lists every word, with a short example of each
          sdata.lesson()           lists the lessons: learn the whole of simplydata in a day
          sdata.example()          lists the made-up tables you can practise with

        You can type the word with or without "chart." or "table." in front and with or
        without brackets: sdata.help("bars"), sdata.help("chart.bars") and sdata.help("bars()") all
        work. A word that charts and tables both have, such as save, explains both;
        sdata.help("table.save") explains only the table one.
        """))


def _print_one(key: str) -> None:
    title = _call_form(key)
    print(title)
    print("=" * len(title))
    print(f"Group: {_group_of(key)}")
    aliases = _aliases().get(key)
    if aliases:
        start = "table" if _kind(key) == "table" else "chart"
        print("Same word, other spelling: " + ", ".join(f"{start}.{a}()" for a in aliases))
    print()
    if key in OPTIONS:
        opt = OPTIONS[key]
        print(opt["about"])
        print()
        print(textwrap.fill(opt["text"], WIDTH))
        print()
        print("Works in: " + ", ".join(f"chart.{w}()" for w in opt["words"]))
    else:
        print(inspect.getdoc(_function(key)) or "")
        print()
        print("What you can put in the brackets (the text above says what each one does):")
        print("\n".join(_options_text(key)))
    if key in EXTRA_NOTES:
        print()
        print(textwrap.fill(EXTRA_NOTES[key], WIDTH))
    print()
    print("Example you can run:")
    print(textwrap.indent(HELP_EXAMPLES[key], "    "))


def help(word=None) -> None:  # noqa: A001 - meant to be sdata.help
    """Explain one word: what it does, its options and an example you can run. sdata.help("bars")

    sdata.help() on its own explains how to get help. Table words work too: sdata.help("keep_rows").
    """
    if word is None:
        _print_help_overview()
        return None
    if isinstance(word, str):
        keys = _lookup_all(word)
    else:
        keys = [k for k in _all_keys() if k not in OPTIONS and _function(k) is word][:1]
        name = getattr(word, "__name__", None)
        if not keys and name is not None:
            keys = _lookup_all(name)
        if not keys:
            return builtins.help(word)  # not a simplydata word: Python's own help
        word = name
    old = _old_names(word)
    # "band=" is an old option (now confidence_band= in fit_line), although band is also a word
    if old and (not keys or str(word).strip().endswith("=")):
        for name, new, key in old:
            where = f" in {_call_form(key)}" if name.endswith("=") else ""
            print(f'"{name}" is now called "{new}"{where}. The old name still works for now, but will '
                  "stop working in a later version.")
        print()
        keys = list(dict.fromkeys(key for _, _, key in old))
    if not keys:
        known = sorted({_normalise(k) for k in _all_keys()} | {a for v in _aliases().values() for a in v})
        lines = [f'simplydata has no word called "{word}".']
        guesses = suggest(_normalise(word), known)
        if guesses:
            lines.append("Did you mean " + " or ".join(f'sdata.help("{g}")' for g in guesses) + "?")
        lines.append("Type sdata.words() to see every word.")
        lines.append(f"Missing something? Ask for it at {ISSUES_URL}")
        print("\n".join(lines))
        return None

    for i, key in enumerate(keys):
        if i:
            print()
            print("-" * WIDTH)
            print()
        _print_one(key)
    if len(keys) > 1:
        forms = " or ".join(f'sdata.help("{_call_form(k)[:-2]}")' for k in keys)
        print()
        print(_wrap(f'"{_normalise(word)}" can mean more than one word, so each one is explained '
                    f"above. For only one, type {forms}."))
    return None


# ---------------------------------------------------------------- sdata.lesson()
def _sx(code: str) -> str:
    """A worked solution's code, written indented in LESSONS and given back without the indent."""
    return textwrap.dedent(code).strip()


# Each lesson: title, what you will learn, one or more parts, and a task.
# Each part: a sentence to start, then (line of code, what it does) pairs.
# The code of each part runs on its own (the tests run every part).
# solutions: one (code, one-line explanation) for each task, in the same order. Each runs on
# its own (the tests run every one), and sdata.lesson(3, solutions=True) prints them.
LESSONS = [
    dict(
        title="Your first chart",
        learn=[
            "how to get a table to practise with",
            "how to start a chart and draw bars",
            "how to save your chart as a picture",
        ],
        parts=[
            ("A simplydata program is made of small steps, one step on each line, and each step "
             "says what it does.", [
                ("import simplydata as sdata", 'Load simplydata. From now on we call it by its short name, "sdata".'),
                ("sdata.example()", "List the made-up tables you can practise with, and their columns."),
                ('marks = sdata.example("marks")', "Get the marks table: test marks of made-up children. "
                                                "We call it marks."),
                ("print(marks)", "Look at the table. Each row is one child in one test. The names at "
                                 "the top (name, class, subject ...) are the columns."),
                ("chart = sdata.chart(marks)", "Start a new, empty chart that uses the marks table. We "
                                            "call it chart, and every next line starts with that name."),
                ('chart.bars(x="name", y="marks")', "Draw bars: one bar for each name (along x, the "
                                                    "bottom), as tall as the marks (up y, the side). "
                                                    "Each child did three tests, so the bar shows the "
                                                    "average: add the three marks up and divide by 3."),
                ('chart.save("lesson1.png")', "Save the chart as a picture called lesson1.png, in the "
                                              "folder you are working in. Open it to see your chart!"),
            ]),
        ],
        notes=[
            "Chart types are nouns on a chart: chart.bars, chart.lines, chart.dots ... The chart is "
            "the thing, and bars is what it shows.",
            'For your own data, write chart = sdata.chart("my_marks.xlsx"). Excel, CSV, TSV, Parquet '
            "and JSON files all work.",
            'chart.save("lesson1.pdf") saves a PDF instead: the end of the name decides the type.',
            "In Jupyter, VS Code or Spyder, write chart on the last line of a cell and the chart "
            "appears by itself. In a script, chart.show() opens it in a window.",
            "Are the names under the bars bumping into each other? Lesson 2 shows how to turn them.",
        ],
        task=[
            'Change x="name" to x="subject". What does each bar show now?',
            'Save the chart as "my_first_chart.png" and also as "my_first_chart.pdf".',
            'Make a chart of the "fruit" table: bars with x="fruit" and y="count".',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="subject", y="marks")
                chart.save("lesson1_subjects.png")"""),
             "Each bar is now one subject, as tall as the average mark of all 18 children in that "
             "subject."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="name", y="marks")
                chart.save("my_first_chart.png")
                chart.save("my_first_chart.pdf")"""),
             "One chart can be saved many times; the end of the file name (.png or .pdf) decides "
             "the type."),
            (_sx("""\
                import simplydata as sdata
                fruit = sdata.example("fruit")
                chart = sdata.chart(fruit)
                chart.bars(x="fruit", y="count")
                chart.save("fruit.png")"""),
             "One bar per fruit. Each fruit was sold several times, so a bar is the average count "
             'of one sale; show="total" would show how many were sold in the whole week.'),
        ],
        words=["sdata.example", "sdata.chart", "bars", "save", "show"],
    ),
    dict(
        title="Titles and labels",
        learn=[
            "how to put a title on top",
            "how to name the x-axis and the y-axis",
            "how to add a small caption and change the size of the text",
        ],
        parts=[
            ("A chart is easier to read when it says what it shows. First draw, then add the words.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.bars(x="name", y="marks")', "Draw one bar for each child."),
                ('chart.title_top("Our class marks")', "Write a title on top of the chart."),
                ('chart.label_x("Student")', "Write a name under the x-axis (the bottom)."),
                ('chart.label_y("Average mark")', "Write a name beside the y-axis (the side)."),
                ("chart.rotate_labels_x(45)", "Turn the names under the bars by 45 degrees, so "
                                              "long names do not bump into each other."),
                ('chart.caption_bottom("Made-up data from sdata.example")', "Write a small line under "
                                                                         "the whole chart."),
                ("chart.font_size(14)", "Make all the text a bit bigger (normal is 12)."),
                ('chart.save("lesson2.png")', "Save it."),
            ]),
            ("Words can also go next to the dots: a name for each dot, where it covers nothing.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('kids = {"name": ["Asha", "Ben", "Chen", "Dina", "Eli"], "hours": [2.0, 5.5, 3.1, 7.2, 4.4], '
                 '"marks": [48, 71, 55, 88, 62]}', "A small table typed by hand: each column name, then its "
                                                   "values in a list."),
                ("chart = sdata.chart(kids)", "Start a chart of it."),
                ('chart.dots(x="hours", y="marks")', "One dot for each child."),
                ('chart.names_on_dots("name")', "Write each name next to its dot, where it covers no other "
                                                "name or dot. A name with no free place is left out, and "
                                                "a line says so."),
                ('chart.label_x("Hours of study")', "Name the x-axis."),
                ('chart.save("lesson2_names.png")', "Save it."),
            ]),
        ],
        notes=[
            "After the chart type, the order of words such as title_top and label_y does not matter: "
            "the chart is drawn when you save or show it.",
            "Words that belong to one axis end in _x or _y: label_x, label_y, rotate_labels_x.",
        ],
        task=[
            'Give the chart your own title, and label the y-axis "Marks out of 100".',
            "Try chart.font_size(20). Then try 9. Which is easier to read?",
            "Take away the line chart.rotate_labels_x(45). What happens to the names?",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="name", y="marks")
                chart.title_top("How well did each child do?")
                chart.label_y("Marks out of 100")
                chart.rotate_labels_x(45)
                chart.save("lesson2_my_title.png")"""),
             "title_top writes any title you like, and label_y names the y-axis."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="name", y="marks")
                chart.title_top("Our class marks")
                chart.rotate_labels_x(45)
                chart.font_size(20)
                chart.save("lesson2_big_text.png")
                chart.font_size(9)
                chart.save("lesson2_small_text.png")"""),
             "The last font_size wins, so one chart gives both pictures. Size 20 is easy to read on "
             "a slide but crowds the names; 9 fits everything but is small. 12 to 14 suits most charts."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="name", y="marks")
                chart.title_top("Our class marks")
                chart.save("lesson2_no_rotate.png")"""),
             "Without rotate_labels_x the 18 names lie flat under the bars and run into each other."),
        ],
        words=["title_top", "label_x", "label_y", "rotate_labels_x", "caption_bottom", "font_size",
               "names_on_dots"],
    ),
    dict(
        title="Colours, groups and legends",
        learn=[
            "how to give each group its own colour",
            "how to choose your own colours",
            "how to move the legend (the box that says which colour is which)",
            "how to change the whole look with a style",
        ],
        parts=[
            ("colour= inside a chart type splits the data into groups.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.bars(x="subject", y="marks", colour="class")', "For each subject, draw one "
                                                                       "bar per class, each class "
                                                                       "in its own colour."),
                ('chart.colours("orange", "teal", "purple")', "Choose the colours yourself, in order: "
                                                              "class A orange, B teal, C purple."),
                ("chart.legend_bottom()", "Put the legend under the chart."),
                ('chart.title_top("Marks in each subject")', "Add a title."),
                ('chart.save("lesson3_groups.png")', "Save it."),
            ]),
            ("Without groups, one colour is used for everything. You can choose it.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.example("weather")', "Get the weather table: three made-up cities."),
                ("chart = sdata.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="rain")', "Draw a line of rain through the months. With "
                                                     "three cities in each month, the line shows "
                                                     "their average."),
                ('chart.colour("blue")', "Make the line blue."),
                ('chart.style("school")', 'Change the whole look. Try "paper", "school", '
                                          '"presentation" and "dark".'),
                ('chart.save("lesson3_look.png")', "Save it."),
            ]),
            ("Groups come in the order they first appear in the table. You can choose another order, "
             "and give the legend a name.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.bars(x="class", y="marks", colour="subject")', "For each class, one bar per subject."),
                ('chart.order_x("C", "B", "A")', "Put the classes along the x-axis in this order."),
                ('chart.order_legend("Science", "Maths", "English")', "Put the subjects in this order: in the "
                                                                      "legend and side by side in each class."),
                ('chart.label_legend("Subject")', "The name above the legend (normally the column name)."),
                ('chart.save("lesson3_order.png")', "Save it."),
            ]),
        ],
        notes=[
            'colour="class" (a column) gives each group a colour. colour="red" (a colour) makes '
            "that step red. color works too.",
            "The legend can go on any side: legend_top, legend_bottom, legend_left, legend_right. "
            "hide_legend takes it away and show_legend brings it back.",
            'Month and weekday names ("Jan", "Monday") come in calendar order by themselves, also in '
            "the small charts of one_chart_per and the frames of animate. order_x(biggest_first=True) "
            "puts the biggest bar first, and order_y orders the groups up the y-axis.",
        ],
        task=[
            'In the weather chart, add colour="city" to chart.lines(...). How many lines do you get?',
            "Move the legend to the top with chart.legend_top().",
            'Try chart.style("dark") and choose three colours that look good on it.',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                weather = sdata.example("weather")
                chart = sdata.chart(weather)
                chart.lines(x="month", y="rain", colour="city")
                chart.save("lesson3_cities.png")"""),
             "Three lines, one for each city, and a legend that says which colour is which city."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="subject", y="marks", colour="class")
                chart.colours("orange", "teal", "purple")
                chart.legend_top()
                chart.save("lesson3_legend_top.png")"""),
             "legend_top puts the legend above the chart, in place of legend_bottom."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="subject", y="marks", colour="class")
                chart.style("dark")
                chart.colours("gold", "turquoise", "hotpink")
                chart.save("lesson3_dark.png")"""),
             "Light, bright colours stand out on a dark background; dark colours such as navy "
             "would almost disappear."),
        ],
        words=["colours", "colour", "legend_bottom", "legend_top", "hide_legend", "style", "lines",
               "order_x", "order_legend", "label_legend"],
    ),
    dict(
        title="Different chart types",
        learn=[
            "lines, dots, histograms, boxes and pies",
            "which chart suits which question",
            "how to put two chart types on top of each other",
        ],
        parts=[
            ("Lines show how something changes, for example over time.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.example("weather")', "Get the weather table."),
                ("chart = sdata.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="temperature", colour="city", dots=True)',
                 "One line per city. dots=True also marks each month with a dot."),
                ('chart.save("lesson4_lines.png")', "Save it."),
            ]),
            ("Dots show if two things go together. This is also called a scatter chart.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks", colour="class")', "One dot for each test: how "
                                                                    "long the child studied, "
                                                                    "and the mark they got."),
                ('chart.save("lesson4_dots.png")', "Save it. Do more hours give more marks?"),
            ]),
            ("A histogram counts how many values fall in each range, like 40 to 50, 50 to 60 ... "
             "Its shape is called the distribution of the values.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.histogram(x="marks", bins=8)', "Cut the marks into 8 ranges (bins) and "
                                                       "draw a bar for how many marks are in each."),
                ('chart.save("lesson4_histogram.png")', "Save it."),
            ]),
            ("Boxes show how spread out the values are. Two chart types on one chart are drawn "
             "one on top of the other.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.boxes(x="class", y="marks")', "One box per class. The line in the box is "
                                                      "the median: the middle mark when they are "
                                                      "put in order. The box holds the middle half "
                                                      "of the marks (the interquartile range)."),
                ('chart.all_dots(x="class", y="marks")', "Also show every single mark as a dot, "
                                                         "on top of the boxes."),
                ('chart.save("lesson4_boxes.png")', "Save it."),
            ]),
            ("A pie shows how a whole is shared out.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('fruit = sdata.example("fruit")', "Get the fruit table: what two shops sold in a week."),
                ("chart = sdata.chart(fruit)", "Start a chart."),
                ('chart.pie(labels="fruit", values="count")', "One slice per fruit, as big as the "
                                                              "number sold."),
                ('chart.save("lesson4_pie.png")', "Save it."),
            ]),
        ],
        notes=[
            "Chart types on the same chart are layers of one chart. Lesson 7 shows how to make "
            "several charts and put them on one page.",
            "There are more chart types: area, violins, heatmap, polar, contour, surface and "
            "dots_3d. Type sdata.words() to see them all and sdata.help(\"violins\") to learn one.",
        ],
        task=[
            'Draw chart.violins(x="class", y="marks") instead of boxes. What is different?',
            'Draw a histogram of the weather "temperature" column with bins=6.',
            'Make a chart of sdata.example("growth") and draw chart.area(x="year", y="population", '
            'colour="country").',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.violins(x="class", y="marks")
                chart.all_dots(x="class", y="marks")
                chart.save("lesson4_violins.png")"""),
             "A violin shows the whole shape of the marks: it is wide where many marks are and "
             "thin where few are. A box shows only the middle half and the median."),
            (_sx("""\
                import simplydata as sdata
                weather = sdata.example("weather")
                chart = sdata.chart(weather)
                chart.histogram(x="temperature", bins=6)
                chart.save("lesson4_temperature.png")"""),
             "Six ranges of temperature, each bar as high as the number of months (of all three "
             "cities) that fall in it."),
            (_sx("""\
                import simplydata as sdata
                growth = sdata.example("growth")
                chart = sdata.chart(growth)
                chart.area(x="year", y="population", colour="country")
                chart.save("lesson4_area.png")"""),
             "An area chart is a line chart with the space under each line filled in."),
        ],
        words=["lines", "dots", "histogram", "boxes", "all_dots", "pie", "violins", "area"],
    ),
    dict(
        title="Statistics: averages, error bars and fit lines",
        learn=[
            "what a bar shows when a group has many values: the average or the median",
            "how to add error bars, and the difference between standard deviation and standard error",
            "how to draw the straight line that fits some dots best",
        ],
        parts=[
            ("The average (also called the mean) is all the values added up, divided by how many "
             "there are. Error bars show how far the values spread around it.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.bars(x="class", y="marks", show="average", error="standard deviation")',
                 "One bar per class, as tall as the average mark. The error bars show the standard "
                 "deviation: how far marks usually are from the average. A line is printed that "
                 "says which error bars were drawn."),
                ("chart.numbers_on_bars()", "Write the average on each bar."),
                ('chart.save("lesson5_average.png")', "Save it."),
            ]),
            ("A fit line is the straight line that goes as close as possible to all the dots.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "One dot per test: hours of study and the mark."),
                ("chart.fit_line(confidence_band=True, show_equation=True)", "Add the best straight "
                 "line (also called linear regression). confidence_band=True shades where the true "
                 "line very likely is (the 95 % confidence interval). show_equation=True writes its "
                 "equation and R² (1 means the dots sit right on the line, 0 means no pattern at all)."),
                ('chart.save("lesson5_fit.png")', "Save it."),
            ]),
            ("A density curve is a smooth histogram: high where many values are.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.density(x="marks", colour="class")', "One smooth curve for each class "
                                                             "(a kernel density estimate)."),
                ('chart.save("lesson5_density.png")', "Save it."),
            ]),
        ],
        notes=[
            "Average or median? The median is the middle value when the values are put in order. "
            "One very big or very small value pulls the average towards it, but hardly moves the "
            "median, so the median is safer when a few values are far from the rest (outliers).",
            'show= can be "average", "median", "total" (all added up), "count" (how many rows), '
            '"smallest", "biggest" or "standard deviation": the same names as the table words '
            "average_of, median_of and so on.",
            "Standard deviation or standard error? The standard deviation describes the values: how "
            "spread out they are. The standard error describes the average: how sure we are of it. "
            "It gets smaller as you collect more values, even when the values spread just as much.",
            'error="confidence interval" draws the 95 % confidence interval: the range the true '
            'average is very likely to be in. The short forms "sd", "se" and "ci" work too. Always '
            "say in your caption which error bars you drew.",
            "fit_curve(degree=2) fits a curve with one bend; smooth_line() follows the dots freely.",
        ],
        task=[
            'Change show="average" to show="median". Do the bars change much?',
            'Change error="standard deviation" to error="standard error". Are the error bars '
            "bigger or smaller? Why?",
            'Make a chart of the weather table, draw dots with x="month_number", y="temperature" '
            "and add chart.fit_curve(degree=2).",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="class", y="marks", show="median")
                chart.numbers_on_bars()
                chart.save("lesson5_median.png")"""),
             "Error bars go with averages, so error= comes out. The medians (A 74, B 68.5, C 57) "
             "are close to the averages (A 71.8, B 64.3, C 58.6): no single mark pulls them far."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="class", y="marks", show="average", error="standard error")
                chart.save("lesson5_standard_error.png")"""),
             "Smaller: the standard error is the standard deviation divided by the square root of "
             "the number of values (here 18), so it is about a quarter of the standard deviation."),
            (_sx("""\
                import simplydata as sdata
                weather = sdata.example("weather")
                chart = sdata.chart(weather)
                chart.dots(x="month_number", y="temperature", colour="city")
                chart.fit_curve(degree=2)
                chart.save("lesson5_fit_curve.png")"""),
             "Each city gets a curve with one bend: warm in the middle of the year, cold at both ends."),
        ],
        words=["show=", "error=", "numbers_on_bars", "fit_line", "density", "fit_curve", "smooth_line"],
    ),
    dict(
        title="Axes and reference lines",
        learn=[
            "how to choose where an axis starts and stops",
            "how to draw a line at a value, like a pass mark, and shade a band",
            "how to choose the tick marks, and what a log scale is",
        ],
        parts=[
            ("The axes are the two edges with numbers: x along the bottom, y up the side.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.bars(x="name", y="marks")', "One bar for each child."),
                ("chart.range_y(0, 100)", "Make the y-axis go from 0 to 100, the lowest and highest "
                                          "mark you can get."),
                ("chart.ticks_y(0, 25, 50, 75, 100)", "Put the numbers on the y-axis at these places."),
                ('chart.line_at_y(50, label="pass mark")', "Draw a dashed line across at 50, with a name."),
                ("chart.shade_between_y(40, 60)", "Shade the band between 40 and 60."),
                ("chart.rotate_labels_x()", "Turn the names so they fit."),
                ('chart.save("lesson6_marks.png")', "Save it."),
            ]),
            ("On a log scale 1, 10, 100 and 1000 are the same distance apart. It helps when some "
             "numbers are much bigger than others.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('growth = sdata.example("growth")', "Get the growth table: people in four made-up "
                                                  "countries."),
                ("chart = sdata.chart(growth)", "Start a chart."),
                ('chart.lines(x="year", y="population", colour="country")', "One line per country."),
                ("chart.log_scale_y()", "Use a log scale on the y-axis. A country that grows by the "
                                        "same share each year now gets a straight line."),
                ('chart.line_at_x(2010, label="2010")', "Draw an up-and-down line at the year 2010."),
                ('chart.label_y("People (millions)")', "Name the y-axis."),
                ('chart.save("lesson6_log.png")', "Save it."),
            ]),
            ("A band shades the space between a low and a high value at each x: a range, or the "
             "uncertainty around a line.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('days = {"day": [1, 2, 3, 4, 5], "lowest": [3, 5, 4, 6, 8], "average": [6, 8, 7, 9, 11], '
                 '"highest": [9, 11, 10, 12, 14]}', "A small table: the lowest, average and highest "
                                                    "temperature on five days."),
                ("chart = sdata.chart(days)", "Start a chart of it."),
                ('chart.band(x="day", between=["lowest", "highest"], label="lowest to highest")',
                 "Shade from the lowest to the highest value of each day. label= names it in the legend."),
                ('chart.lines(x="day", y="average")', "The average as a line through the band."),
                ('chart.label_y("Temperature (°C)")', "Name the y-axis."),
                ('chart.save("lesson6_band.png")', "Save it."),
            ]),
        ],
        notes=[
            "Every axis word ends in _x or _y: range_x, ticks_x, log_scale_x, line_at_x, "
            "shade_between_x, flip_x ...",
            "range_y(0, None) fixes only the bottom and lets the top choose itself.",
            "Bars should start at 0: a bar is read by its length, so a y-axis that starts at 40 "
            "makes small differences look big.",
        ],
        task=[
            "Move the pass mark to 60 and shade between 0 and 40 instead.",
            "Take away chart.log_scale_y(). Which countries look different?",
            "Try chart.flip_y() on any chart. What happens?",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="name", y="marks")
                chart.range_y(0, 100)
                chart.line_at_y(60, label="pass mark")
                chart.shade_between_y(0, 40)
                chart.rotate_labels_x()
                chart.save("lesson6_pass_60.png")"""),
             "line_at_y(60) moves the line; shade_between_y(0, 40) shades the band from 0 to 40."),
            (_sx("""\
                import simplydata as sdata
                growth = sdata.example("growth")
                chart = sdata.chart(growth)
                chart.lines(x="year", y="population", colour="country")
                chart.line_at_x(2010, label="2010")
                chart.label_y("People (millions)")
                chart.save("lesson6_no_log.png")"""),
             "The lines now bend upwards instead of running straight, and the small countries "
             "(Cherria and Applia) sit low and look flat, although Cherria nearly triples."),
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.bars(x="class", y="marks")
                chart.flip_y()
                chart.save("lesson6_flip.png")"""),
             "The y-axis turns upside down: 0 is at the top and the bars hang down. Useful for "
             "depths, where bigger numbers are further down."),
        ],
        words=["range_y", "ticks_y", "line_at_y", "shade_between_y", "log_scale_y", "line_at_x", "flip_y",
               "band"],
    ),
    dict(
        title="Several charts together",
        learn=[
            "how to make one small chart for each group",
            "how to put different charts on one page: side by side, one above the other, or in a grid",
            "how to write A, B, C on the charts, as figures in papers have",
            "how to compare every number column with every other",
        ],
        parts=[
            ("one_chart_per draws the same chart once for each group.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.fit_line()", "Add a fit line."),
                ('chart.one_chart_per("class")', "Make one small chart for each class, with the same "
                                                 "axes so they are easy to compare."),
                ('chart.save("lesson7_one_chart_per.png")', "Save it."),
            ]),
            ("Each chart is an object of its own, so different charts can share one page.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.example("weather")', "Get the weather table."),
                ('a = sdata.chart(weather).lines(x="month", y="temperature", colour="city")',
                 "Make a first chart, called a. Steps can also be joined with dots on one line."),
                ('b = sdata.chart(weather).bars(x="month", y="rain", colour="city")',
                 "Make a second chart, called b."),
                ("page = sdata.one_above_other(a, b)", "Put a above b on one page."),
                ('page.title_top("A year of weather")', "Give the whole page a title."),
                ('page.save("lesson7_page.png")', "Save the page."),
            ]),
            ("A grid holds many charts in rows and columns. letters=True writes A, B, C and D on "
             "them, as figures in papers have.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ('a = sdata.chart(marks).bars(x="class", y="marks").title_top("Average")', "Chart a."),
                ('b = sdata.chart(marks).histogram(x="marks").title_top("All marks")', "Chart b."),
                ('c = sdata.chart(marks).dots(x="height", y="marks").title_top("Height")', "Chart c."),
                ('d = sdata.chart(marks).boxes(x="subject", y="marks").title_top("Subjects")', "Chart d."),
                ("page = sdata.grid(a, b, c, d, per_row=2, letters=True)", "All four in a grid, 2 charts "
                                                                        "per row, with a letter on each."),
                ('page.save("lesson7_grid.png")', "Save it."),
            ]),
            ("Charts from different tables can share a page too.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('rain = sdata.chart(sdata.example("weather")).lines(x="month", y="rain", colour="city")',
                 "A chart of the weather table, called rain."),
                ('forest = sdata.chart(sdata.example("growth")).lines(x="year", y="forest", colour="country")',
                 "A chart of the growth table, called forest."),
                ("page = sdata.side_by_side(rain, forest, letters=True)", "Put the two charts next to "
                                                                       "each other, with A and B."),
                ('page.save("lesson7_two_tables.png")', "Save the page."),
            ]),
            ("sdata.pairs draws every number column against every other, in one go.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('pairs = sdata.pairs(sdata.example("marks"), colour="class")', "Hours, marks and height, "
                                                                          "each against each other. "
                                                                          "The middle line shows how "
                                                                          "each column is spread out."),
                ('pairs.save("lesson7_pairs.png")', "Save it. Which two columns go together?"),
            ]),
        ],
        notes=[
            "same_y=True gives all the charts on a page the same y-axis, so heights can be "
            "compared fairly: sdata.side_by_side(a, b, same_y=True).",
            "sdata.side_by_side, sdata.one_above_other and sdata.grid all take letters=True.",
            'chart.one_chart_per(down="class", across="subject") makes a table of small charts: '
            "the classes go down, the subjects go across. per_row=3 puts 3 small charts in a row.",
        ],
        task=[
            'Change chart.one_chart_per("class") to chart.one_chart_per("subject").',
            "Use sdata.side_by_side(a, b) instead of sdata.one_above_other(a, b) for the weather charts.",
            'Make sdata.pairs of sdata.example("weather") with colour="city".',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                marks = sdata.example("marks")
                chart = sdata.chart(marks)
                chart.dots(x="hours", y="marks")
                chart.fit_line()
                chart.one_chart_per("subject")
                chart.save("lesson7_per_subject.png")"""),
             "One small chart for each subject (Maths, Science and English), each with its own fit line."),
            (_sx("""\
                import simplydata as sdata
                weather = sdata.example("weather")
                a = sdata.chart(weather).lines(x="month", y="temperature", colour="city")
                b = sdata.chart(weather).bars(x="month", y="rain", colour="city")
                page = sdata.side_by_side(a, b)
                page.title_top("A year of weather")
                page.save("lesson7_side_by_side.png")"""),
             "The same two charts, now next to each other instead of one above the other."),
            (_sx("""\
                import simplydata as sdata
                pairs = sdata.pairs(sdata.example("weather"), colour="city")
                pairs.save("lesson7_weather_pairs.png")"""),
             "Every number column of the weather table against every other, one colour per city."),
        ],
        words=["one_chart_per", "sdata.one_above_other", "sdata.grid", "sdata.side_by_side", "sdata.pairs",
               "fit_line", "histogram"],
    ),
    dict(
        title="Hard tricks, and your own program",
        learn=[
            "a second y-axis, a zoom box and a moving chart",
            "a heatmap that sorts itself",
            "how to write your own program from start to end",
        ],
        parts=[
            ("A second y-axis shows a column measured in other units on the same chart.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.example("weather")', "Get the weather table."),
                ("chart = sdata.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="temperature")', "A line of temperature (average of "
                                                            "the three cities)."),
                ('chart.second_y("rain", drawn_as="bars", label="Rain (mm)")', "Add rain, drawn as "
                                                                               "bars, with its own "
                                                                               "y-axis on the right."),
                ('chart.label_y("Temperature (°C)")', "Name the left y-axis."),
                ('chart.save("lesson8_second_y.png")', "Save it."),
            ]),
            ("A zoom box shows a small part of the chart close up.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('marks = sdata.example("marks")', "Get the marks table."),
                ("chart = sdata.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.zoom_box(x_between=[2, 4], y_between=[40, 60], where=\"top left\")",
                 "Show the part between 2 and 4 hours and between 40 and 60 marks, bigger, in a "
                 "box at the top left."),
                ('chart.save("lesson8_zoom.png")', "Save it."),
            ]),
            ("A heatmap colours a grid of numbers. cluster=True puts rows that are alike next to "
             "each other.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.example("weather")', "Get the weather table."),
                ("chart = sdata.chart(weather)", "Start a chart."),
                ('chart.heatmap(x="month", y="city", value="rain", numbers=True, cluster=True)',
                 "One square per city and month, coloured by rain, with the number written in. "
                 "cluster=True sorts the rows and columns and draws a tree of how alike they are "
                 "(hierarchical clustering)."),
                ('chart.label_colour_bar("Rain (mm)")', "Name the colour bar."),
                ("chart.size(width=12, height=4)", "Make the chart wide and low."),
                ('chart.save("lesson8_heatmap.png")', "Save it."),
            ]),
            ("An animation is a chart that moves: one picture for each value of a column.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('growth = sdata.example("growth")', "Get the growth table."),
                ("chart = sdata.chart(growth)", "Start a chart."),
                ('chart.bars(x="country", y="population")', "One bar per country."),
                ("chart.range_y(0, 50)", "Keep the y-axis still, from 0 to 50."),
                ('chart.animate(over="year")', "Make one picture for each year."),
                ('chart.save("lesson8_growth.gif")', "Save it as a .gif, a moving picture. Open "
                                                     "it in a web browser to watch."),
            ]),
            ("Your own program. Every chart follows the same plan: get the data, look, start, draw, "
             "words, look, save.", [
                ("import simplydata as sdata", "1. Load simplydata."),
                ('island = sdata.example("island")', "2. Get the data. For your own file, skip this line "
                                                  'and write chart = sdata.chart("my_file.xlsx") below.'),
                ("print(island)", "3. Look at the data before you draw it."),
                ("chart = sdata.chart(island)", "4. Start a chart."),
                ('chart.contour(x="east", y="north", z="height")', "5. Draw: a map where each "
                                                                   "colour band is one height."),
                ('chart.title_top("Treasure Island")', "6. Words: a title ..."),
                ('chart.label_x("East (km)")', "... and the axis names ..."),
                ('chart.label_y("North (km)")', ""),
                ('chart.label_colour_bar("Height (m)")', "... and the name of the colour bar."),
                ('chart.note("X marks the spot", x=10, y=4, arrow=True)', "A note with an arrow."),
                ('chart.style("presentation")', "7. Look: big and clear for showing to others."),
                ('chart.save("lesson8_my_program.png")', "8. Save."),
            ]),
        ],
        notes=[
            "Those are all the charts! sdata.words() lists every word and sdata.help(\"word\") "
            "explains any of them. Lessons 9 to 12 teach tables: how to look at your data, clean "
            "it and sum it up before you draw it.",
            'Your own data: save it from Excel as .xlsx or .csv, then chart = sdata.chart("my_file.xlsx"). '
            "If a column name is spelt wrong, simplydata tells you the right one.",
        ],
        task=[
            "Write your own program, from an empty file, for a question you care about.",
            "Use any example table, or your own. Give it a title, axis names and a note.",
            "Make a second chart of the same table, put the two charts side by side and save the "
            "page as a PDF.",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                fruit = sdata.example("fruit")
                print(fruit)
                chart = sdata.chart(fruit)
                chart.bars(x="fruit", y="count", show="total")
                chart.numbers_on_bars()
                chart.title_top("Which fruit sells best?")
                chart.label_x("Fruit")
                chart.label_y("Sold in one week")
                chart.save("lesson8_fruit.png")"""),
             "One possible question: which fruit sells best? Same plan as always: get the data, "
             "look, start, draw, words, save."),
            (_sx("""\
                import simplydata as sdata
                growth = sdata.example("growth")
                chart = sdata.chart(growth)
                chart.lines(x="year", y="forest", colour="country")
                chart.title_top("Forest in four countries")
                chart.label_x("Year")
                chart.label_y("Forest (% of the land)")
                chart.note("Applia loses the most", x=2016, y=50, arrow=True)
                chart.save("lesson8_forest.png")"""),
             "Another table, with a title, both axis names and a note with an arrow."),
            (_sx("""\
                import simplydata as sdata
                growth = sdata.example("growth")
                people = sdata.chart(growth).lines(x="year", y="population", colour="country")
                people.label_y("People (millions)")
                forest = sdata.chart(growth).lines(x="year", y="forest", colour="country")
                forest.label_y("Forest (% of the land)")
                page = sdata.side_by_side(people, forest, letters=True)
                page.save("lesson8_page.pdf")"""),
             "Two charts of one table on one page, saved as a PDF because the name ends in .pdf."),
        ],
        words=["second_y", "zoom_box", "heatmap", "label_colour_bar", "size", "animate",
               "contour", "note", "style", "sdata.words", "sdata.help"],
    ),
    dict(
        title="Tables: open one and look at it",
        learn=[
            "how to turn data into a simplydata table",
            "how to look at the first and last rows, and the names of the columns",
            "how to describe every column and count what is in it",
        ],
        parts=[
            ("Before you draw, look at your data. sdata.table gives you plain words for that.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table from the practice marks. For "
                                                          'your own file, write sdata.table("my_file.xlsx").'),
                ("print(table)", "Print the table. The first line says how many rows and columns it "
                                 "has. Then come the first 10 rows."),
                ("table.show_first_rows(5)", "Show only the first 5 rows."),
                ("table.show_last_rows(3)", "Show the last 3 rows."),
                ("print(table.column_names())", "Print the names of the columns."),
                ("table.describe_columns()", "Describe every column: is it numbers or text, how many "
                                             "cells are empty, and the smallest, biggest and average "
                                             "number."),
                ("table.count_rows()", "Count the rows, and say the answer in a sentence. One row is "
                                       "one child in one test."),
            ]),
            ("count_each counts how often each different value comes up.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("fruit"))', "Make a table of fruit sold by two shops."),
                ("table.show_first_rows()", "Show the first rows. With nothing in the brackets you "
                                            "get 5."),
                ('table.count_each("fruit")', "For each fruit: in how many rows it comes up. It says "
                                              "the answer in a sentence, and gives back a small table "
                                              "that also has the percent of all the rows."),
                ('table.count_each("shop")', "The same for each shop."),
            ]),
            ("sdata.table can read many files at once. Here we first save two small files, then "
             "read them back together.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.table(sdata.example("weather"))', "Make a table of the weather."),
                ("weather.keep_first_rows(12)", "Keep the first 12 rows: one city, a whole year."),
                ('weather.save("lesson9_part1.csv")', "Save them as a CSV file."),
                ("weather.start_again()", "Back to the whole table."),
                ("weather.keep_last_rows(12)", "Keep the last 12 rows: another city."),
                ('weather.save("lesson9_part2.csv")', "Save them as a second file."),
                ('both = sdata.table("lesson9_part*.csv")', "Read every file whose name fits the pattern "
                                                            "(* stands for any text) and stack them. A new "
                                                            'column, "file", says where each row came from.'),
                ('both.count_each("file")', "12 rows from each file."),
                ("both.show(rows=3)", "Show the first 3 rows with every column. both.show() shows the "
                                      "whole table."),
            ]),
        ],
        notes=[
            "Looking never changes the table. Only steps that say what they change (keep_rows, "
            "sort_by ...) do, and lesson 10 shows them.",
            "In Jupyter, VS Code or Spyder, write table on the last line of a cell to see it as a "
            "nice table.",
            "sdata.table reads the same files as sdata.chart: .xlsx, .csv, .tsv, .parquet and .json. "
            'For one sheet of an Excel file: sdata.table("results.xlsx", sheet="2026").',
            "The row numbers on the left start at 0, as everywhere in Python.",
            'A folder works too: sdata.table("data/") reads every table file in it. A web address '
            'works like a file name: sdata.table("https://example.org/data.csv").',
            'Files written in another way: separator=";", decimal_mark="," (3,5 is three and a half), '
            'thousands_mark=".", empty_values=["NA", "-"], skip_rows=2 and encoding="latin-1".',
        ],
        task=[
            'Make a table of sdata.example("weather"). How many rows does it have?',
            "Describe its columns. Which column has the biggest numbers?",
            'Count each "city". Does every city have the same number of rows?',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.count_rows()"""),
             "36 rows: 12 months for each of 3 cities."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.describe_columns()"""),
             "wind_direction has the biggest numbers (up to 354): it is an angle in degrees, from 0 "
             "to 360."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.count_each("city")"""),
             "Yes: Frostby, Rainbridge and Sunnyvale have 12 rows each, one for every month."),
        ],
        words=["sdata.table", "table.show_first_rows", "table.show_last_rows", "table.column_names",
               "table.describe_columns", "table.count_rows", "table.count_each", "table.show"],
    ),
    dict(
        title="Tables: choose rows and columns, and undo",
        learn=[
            "how to keep only the columns and rows you want",
            "how to sort the rows",
            "how to take a step back, and start again",
            "how to save your table",
        ],
        parts=[
            ("A table is changed in small steps, like a chart. Each step says what it did.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.keep_columns("name", "subject", "marks")', "Keep only these three columns, "
                                                                   "in this order."),
                ('table.keep_rows("subject", equal_to="Maths")', "Keep only the rows where the subject "
                                                                 "is Maths. simplydata says how many "
                                                                 "rows it kept."),
                ('table.keep_rows("marks", above=60)', "Of those, keep only marks above 60."),
                ('table.sort_by("marks", biggest_first=True)', "Put the rows in order, the biggest "
                                                               "mark first."),
                ("table.show_first_rows(10)", "Show the first 10 rows: the best Maths marks."),
                ("table.steps()", "List every step you did, in order."),
                ("table.undo()", "Take back the last step (the sorting)."),
                ("table.start_again()", "Go back to the whole table, as it was at the start."),
                ("print(table)", "All 54 rows and 6 columns are back."),
            ]),
            ("keep_rows and remove_rows know many conditions. Then save what you made.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("weather"))', "Make a table of the weather."),
                ('table.keep_rows("month_number", between=[6, 8])', "Keep the summer months: 6, 7 and "
                                                                     "8. between= keeps both ends."),
                ('table.remove_rows("city", equal_to="Frostby")', "Remove the rows of the city "
                                                                  "Frostby."),
                ('table.remove_columns("wind_direction", "wind_speed")', "Remove two columns we do "
                                                                         "not need."),
                ('table.rename_column("temperature", to="temp")', 'Give the column "temperature" '
                                                                  'the shorter name "temp".'),
                ("print(table)", "Look at what is left: 6 rows and 5 columns."),
                ('table.save("lesson10_summer.xlsx")', "Save the table as an Excel file. A name "
                                                       "ending in .csv saves a CSV file."),
            ]),
            ("A condition can also be written out, with and, or, not and brackets. And keep_biggest_rows "
             "keeps the top rows of each group.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ("table.keep_rows(\"marks > 60 and subject == 'Maths'\")",
                 "Keep the rows where both are true: a mark above 60, in Maths. Text goes in quotes, "
                 "and == means \"is equal to\"."),
                ("table.undo()", "Back to all the rows."),
                ("table.remove_rows(\"hours < 3 or class in ['B', 'C']\")",
                 "Remove the rows where at least one is true: under 3 hours of study, or class B or C."),
                ("table.start_again()", "Back to the whole table."),
                ('table.keep_biggest_rows(2, by="marks", for_each="subject")', "Keep the 2 best marks in "
                                                                              "each subject: 6 rows."),
                ("table.show()", "Show the whole (small) table."),
            ]),
        ],
        notes=[
            "Conditions for keep_rows and remove_rows: above, below, at_least, at_most, equal_to, "
            "not_equal_to, between, one_of, contains, starts_with, ends_with and is_empty.",
            'one_of= takes a list: table.keep_rows("subject", one_of=["Maths", "Science"]).',
            "undo() takes back one step at a time. start_again() is a step too, so undo() can take "
            "it back.",
            "Your file is never changed: only table.save(...) writes a file.",
            "Written conditions use == != < <= > >=, and, or, not, brackets, in [...] and is empty. "
            "An empty cell never meets a condition, and not does not change that: \"not (marks > 50)\" "
            "leaves out empty marks too. To find them, write \"marks is empty\". A note says how many "
            "rows had an empty cell.",
            'Dates go in quotes as year-month-day: table.keep_rows("date > \'2026-03-01\'"), or '
            'table.keep_rows("date", after="2026-03-01").',
        ],
        task=[
            'In the marks table, keep only class "B" and sort by "hours". Who studied the longest?',
            "Undo the sorting, then sort by name instead.",
            'Keep 5 random rows with table.keep_random_rows(5, seed=1) and save them as "five.csv".',
            "With one written condition, keep the rows with marks below 50 or hours above 8. How many "
            "rows are there?",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.keep_rows("class", equal_to="B")
                table.sort_by("hours", biggest_first=True)
                table.show_first_rows(3)"""),
             "Jade studied the longest: 9.4 hours, for Maths. biggest_first=True puts the longest on top."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.keep_rows("class", equal_to="B")
                table.sort_by("hours", biggest_first=True)
                table.undo()
                table.sort_by("name")
                table.show_first_rows()"""),
             "undo() takes back only the sorting; class B stays. Text is sorted from A to Z."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.keep_random_rows(5, seed=1)
                print(table)
                table.save("five.csv")"""),
             "seed=1 picks the same 5 rows every time, so someone else can repeat it."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.keep_rows("marks < 50 or hours > 8")
                table.count_rows()"""),
             "19 rows: or keeps a row when at least one of the two is true."),
        ],
        words=["table.keep_columns", "table.keep_rows", "table.sort_by", "table.steps", "table.undo",
               "table.start_again", "table.remove_rows", "table.remove_columns", "table.rename_column",
               "table.save", "table.keep_random_rows", "table.keep_biggest_rows"],
    ),
    dict(
        title="Tables: clean the data and add new columns",
        learn=[
            "how to find and fill empty cells",
            "how to fix columns: numbers written as text, extra spaces, wrong values",
            "how to add new columns, in plain words or like a sum in maths",
            "what long and wide tables are",
        ],
        parts=[
            ("Real data often has empty cells. The practice tables have none, so we make some first.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.replace_values("marks", [61, 73], with_value=None)', "Make every mark of 61 "
                                                                             "or 73 empty. "
                                                                             "with_value=None means "
                                                                             "an empty cell."),
                ("table.count_empty_cells()", "Show how many empty cells each column has."),
                ('table.fill_empty_cells("marks", with_the="average")', "Fill the empty marks with the "
                                                                        "average of the other marks."),
                ("table.undo()", "Changed your mind? Take it back: the cells are empty again."),
                ('table.remove_rows_with_empty_cells("marks")', "This time remove the rows with an "
                                                                "empty cell in marks."),
                ("print(table)", "Some rows fewer than at the start."),
            ]),
            ("Data typed by people is often untidy. These words fix whole columns at once.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table({"name": ["  Asha", "ben ", "Chen"], "height": ["144", "152", "unknown"]})',
                 "Make a small table by hand: each column name, then its values in a list."),
                ('table.remove_extra_spaces("name")', "Take away the spaces before and after the names."),
                ('table.replace_values("name", "ben", with_value="Ben")', 'Write "ben" with a capital '
                                                                          "letter."),
                ('table.make_numbers("height")', "Turn the heights from text into numbers. "
                                                 '"unknown" is not a number, so that cell becomes '
                                                 "empty, and simplydata tells you."),
                ("table.describe_columns()", "Now height is a column of numbers."),
            ]),
            ("A new column can be worked out from other columns, in plain words.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.add_column("marks_per_hour", by_dividing=["marks", "hours"])',
                 'A new column called "marks_per_hour": the marks divided by the hours.'),
                ('table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], '
                 'names=["D", "C", "B", "A"])',
                 "A grade for each mark: below 40 is D, 40 to 59 is C, 60 to 79 is B, 80 and more is A."),
                ("table.show_first_rows()", "Look at the two new columns on the right."),
                ('table.count_each("grade")', "How many of each grade?"),
            ]),
            ("Or write the new column like a sum in maths. Here each subject first gets its own "
             "column: the table goes from long to wide.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.keep_columns("name", "class", "subject", "marks")', "Keep the columns we need."),
                ('table.turn_rows_into_columns(names_from="subject", values_from="marks")',
                 "Make one column for each subject: Maths, Science and English. Each child now has "
                 "one row."),
                ('table.add_column("total", "Maths + Science + English")', 'A new column "total": '
                                                                           "the three marks added up."),
                ('table.add_column("average", "total / 3")', "The average of the three marks."),
                ('table.add_rank_column("place", from_column="total")', "Each child's place: 1 for "
                                                                        "the biggest total."),
                ('table.sort_by("place")', "Put the rows in order of place."),
                ("table.show_first_rows()", "Who came first?"),
            ]),
            ("The shop table has real dates and some empty cells: days when the till broke. Look "
             "where they are, then fill them.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("shop"))', "Daily sales of two shops, January to "
                                                             "March 2026."),
                ("table.count_empty_cells()", "10 empty sales and 5 empty customer counts."),
                ("look = table.chart()", "A chart of the table, to see where the empty cells are."),
                ("look.empty_cells()", "One strip per column and one line per row: empty cells are "
                                       "coloured, and each column says how many it has."),
                ('look.save("lesson11_empty_cells.png")', "Save it."),
                ('table.add_is_empty_column("sales_was_empty", from_column="sales")',
                 "Before filling: a True/False column that remembers which sales were empty."),
                ('table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date", '
                 'for_each="shop")',
                 "Fill each gap on the straight line between the day before and the day after, in date "
                 "order, for each shop on its own. A gap at the very start or end has a neighbour on one "
                 "side only, so it stays empty, and a note says so."),
                ('table.add_date_part_column("weekday", from_column="date", part="weekday name")',
                 "A new column with the weekday of each date."),
                ('table.add_rolling_column("sales_7_days", from_column="sales", window=7, for_each="shop", '
                 'sorted_by="date")',
                 "The average of each day and the 6 days before it: a 7-day moving average, which "
                 "smooths out the busy weekends."),
                ("table.show(rows=8)", "Look at the first 8 rows."),
            ]),
            ("Names and dates typed in by people often need taking apart.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table({"name": ["Asha Rao", "Ben Lee", "Chen Wu"], '
                 '"born": ["04/30/2014", "12/25/2013", "07/03/2014"], "height": [144.37, 152.81, 139.5]})',
                 "A small table typed by hand. The birthdays are written month first, as in the USA."),
                ('table.split_column("name", separator=" ", into=["first_name", "family_name"])',
                 "Split each name at the space into two new columns."),
                ('table.add_text_part_column("initial", from_column="first_name", first_characters=1)',
                 "The first letter of each first name."),
                ('table.replace_text("family_name", "Lee", with_value="Li")', "Replace a piece of text "
                                                                              "inside the cells."),
                ('table.make_dates("born", order="month day year")',
                 "Turn the birthdays into dates. order= says how they are written, so 07/03/2014 is read "
                 "as 3 July and not as 7 March."),
                ('table.add_date_part_column("month_born", from_column="born", part="month name")',
                 "The month of each birthday, as a word."),
                ('table.change_column("height", "height / 100")', "Change the heights from centimetres "
                                                                  "to metres, in the same column."),
                ('table.round_to("height", 2)', "Round them to 2 decimal places."),
                ("table.show()", "Look at the whole table."),
            ]),
        ],
        notes=[
            "Long or wide? A long table has one row for each measurement: the marks table has a row "
            "for each child in each subject. A wide table puts the measurements of one child side "
            "by side, one column per subject. Charts and tests like long tables (also called tidy "
            "data); sums across columns, like the total above, need wide ones.",
            "Filling empty cells is called imputation. Filling with the average makes the values "
            "look less spread out than they really are, so say in your report how you filled them. "
            'with_the="median", "value above" or "value below" and with_value=0 work too.',
            "remove_rows_with_empty_cells() with nothing in the brackets removes every row that has "
            "an empty cell in any column.",
            "More fixing words: make_text, make_dates, make_lowercase and make_uppercase.",
            "Plain words for new columns: by_adding=, by_subtracting=, by_multiplying= and "
            "by_dividing=, or from_column= with plus=, minus=, times= or divide_by=.",
            'In a maths-style column, a column name with spaces goes in backticks: "`max temp` - 2".',
            'with_the="line between neighbours" is linear interpolation. It suits values that change '
            "smoothly, like sales from day to day; it cannot know about a sudden jump. Keep a True/False "
            "column of what you filled, and say in your report how you filled it.",
            "Charts say it too: when a chart leaves out rows because of empty cells, one line tells you "
            "how many.",
            "Row after row: add_previous_value_column gives the value of the row before, and "
            "add_change_column the change since then (in_percent=True for a percent).",
        ],
        task=[
            "In the first example, fill the empty marks with the median instead of the average.",
            'Add a column "height_m" with the height in metres: from_column="height", divide_by=100.',
            'In the last example, add a column "best_two" with the formula "total - English".',
            'In the shop example, add a column "change" with the change in sales from the day before, '
            'for_each="shop" and sorted_by="date" (add_change_column).',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.replace_values("marks", [61, 73], with_value=None)
                table.count_empty_cells()
                table.fill_empty_cells("marks", with_the="median")
                table.count_empty_cells()"""),
             'with_the="median" fills each empty mark with the middle mark of the others; afterwards '
             "no cell is empty."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.add_column("height_m", from_column="height", divide_by=100)
                table.show_first_rows()"""),
             "The height in centimetres divided by 100: 144 becomes 1.44."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.keep_columns("name", "class", "subject", "marks")
                table.turn_rows_into_columns(names_from="subject", values_from="marks")
                table.add_column("total", "Maths + Science + English")
                table.add_column("best_two", "total - English")
                table.show_first_rows()"""),
             "The formula can use a column made earlier (total); best_two is Maths + Science."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("shop"))
                table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date", for_each="shop")
                table.add_change_column("change", from_column="sales", for_each="shop", sorted_by="date")
                table.show(rows=5)"""),
             "Each change is today's sales minus yesterday's, in the same shop; the first day of each "
             "shop has nothing before it, so its change is empty."),
        ],
        words=["table.replace_values", "table.count_empty_cells", "table.fill_empty_cells",
               "table.remove_rows_with_empty_cells",
               "table.remove_extra_spaces", "table.make_numbers", "table.add_column",
               "table.add_group_column", "table.add_rank_column", "table.turn_rows_into_columns",
               "table.add_is_empty_column", "empty_cells", "table.add_date_part_column",
               "table.add_rolling_column", "table.split_column", "table.add_text_part_column",
               "table.replace_text", "table.make_dates", "table.change_column", "table.round_to",
               "table.add_change_column"],
    ),
    dict(
        title="Tables: summaries, and from a table to a chart",
        learn=[
            "how to work out averages, totals and counts, for each group",
            "how to tell if two columns go together, and why that is not the same as a cause",
            "how to draw a table, and which rows a chart uses",
            "how to join two tables and turn columns into rows",
        ],
        parts=[
            ("A summary turns many rows into a few numbers. It never changes your table, and it "
             "says its answer in a sentence.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.average_of("marks")', "The average of all the marks: just one number."),
                ('table.average_of("marks", for_each="class")', "One average for each class. for_each= "
                                                                "also gives back a new, small table."),
                ('table.biggest_of("marks", for_each="subject")', "The best mark in each subject."),
                ('table.correlation_between("hours", "marks")', "Do more hours go with more marks? "
                                                                "The answer is a number r from -1 "
                                                                "to 1 (Pearson's correlation), "
                                                                "explained in words."),
            ]),
            ("A summary for each group is a table too, so you can sort it and draw it.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("fruit"))', "Make a table of fruit sales."),
                ('totals = table.total_of("count", for_each="fruit")', "A new table called totals: one "
                                                                       "row per fruit, with a column "
                                                                       '"total of count". table itself '
                                                                       "does not change."),
                ('totals.sort_by("total of count", biggest_first=True)', "Best seller first."),
                ("chart = totals.chart()", "Start a chart from the summary table."),
                ('chart.bars(x="fruit", y="total of count")', "One bar per fruit."),
                ("chart.numbers_on_bars()", "Write the total on each bar."),
                ('chart.save("lesson12_fruit.png")', "Save the chart."),
            ]),
            ("A chart uses the table as it was when you made the chart. Table steps after it are "
             "for the next chart.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.keep_rows("class", equal_to="A")', "Keep class A."),
                ("all_tests = table.chart()", "A chart of the table as it is now: every test of "
                                              "class A. sdata.chart(table) does the same."),
                ('all_tests.dots(x="hours", y="marks")', "Draw hours against marks."),
                ('table.keep_rows("subject", equal_to="Maths")', "A table step after the chart. The "
                                                                 "chart all_tests does not change."),
                ("maths = table.chart()", "A second chart, of the table as it is now: only the Maths "
                                          "tests of class A."),
                ('maths.dots(x="hours", y="marks")', "Draw the same thing again."),
                ("page = sdata.side_by_side(all_tests, maths, same_y=True)", "Both charts on one page, "
                                                                          "with the same y-axis, so "
                                                                          "you can compare them."),
                ('page.save("lesson12_maths.png")', "Save the page."),
            ]),
            ("Two tables can be joined: their rows are matched on a column they share.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('ages = {"name": ["Asha", "Ben", "Chen"], "age": [12, 13, 12]}', "A small table of "
                                                                                  "ages, typed by hand: "
                                                                                  "each column name, "
                                                                                  "then its values."),
                ('table.join_with(ages, using="name")', "Bring in the age, matching the rows on the "
                                                        "name. Only children found in both tables stay."),
                ("table.show_first_rows()", "Look: there is a new column, age."),
                ('table.average_of("marks", for_each="age")', "The average mark at each age."),
            ]),
            ("Turning columns into rows makes a wide table long: the shape most charts like.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table({"name": ["Asha", "Ben", "Chen"], "test_1": [61, 73, 81], '
                 '"test_2": [70, 68, 90]})', "A small, wide table: two tests, one column for each."),
                ('table.turn_columns_into_rows(["test_1", "test_2"], names_to="test", values_to="mark")',
                 'Put the two tests under each other. The new column "test" says which test, and '
                 '"mark" holds the mark. Each child now has two rows.'),
                ("table.show_first_rows(6)", "Look at the new, long shape."),
                ("chart = table.chart()", "Start a chart from it."),
                ('chart.bars(x="name", y="mark", colour="test")', "Both tests side by side for each "
                                                                  "child."),
                ('chart.save("lesson12_tests.png")', "Save it."),
            ]),
            ("summary_of gives several measures at once, and every= sums up dates by week, month or "
             "year.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("shop"))', "Daily sales of two shops."),
                ('table.summary_of("sales", for_each="shop", show=["average", "median", "standard deviation", '
                 '"count", "empty cells"])',
                 "Five measures of each shop's sales in one tidy table. count counts the days with a "
                 "number; empty cells counts the others."),
                ('table.total_of("sales", for_each="date", every="month")', "The total sales of each "
                                                                            "month: every= puts the days "
                                                                            "of one month together."),
                ('table.percent_of("sales", above=500, for_each="shop")',
                 "On what percent of the days did each shop take more than 500? Each percent comes with "
                 "its 95 % confidence interval."),
            ]),
            ("A cross table puts one group down the side and another across the top. A correlation "
             "table compares many columns at once.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('fruit = sdata.table(sdata.example("fruit"))', "A table of fruit sales."),
                ('fruit.count_rows(for_each="shop", across="fruit")', "How many sales of each fruit in "
                                                                      "each shop: one row per shop, one "
                                                                      "column per fruit."),
                ('fruit.count_different_values("fruit", for_each="shop")', "How many different fruits "
                                                                           "each shop sold."),
                ('marks = sdata.table(sdata.example("marks"))', "A table of the marks."),
                ('marks.average_of("marks", for_each="class", across="subject")', "The average mark of "
                                                                                  "each class in each "
                                                                                  "subject."),
                ('marks.correlation_between("hours", "marks", "height")', "Every pair of the three "
                                                                         "columns at once: a "
                                                                         "correlation table, with the "
                                                                         "strongest pairs in words."),
            ]),
        ],
        notes=[
            "Correlation is not cause. Children who study more might also sleep more, or like the "
            "subject more; r only says that two columns go up and down together, not which one "
            "makes the other change.",
            "More summaries: median_of, total_of, smallest_of, biggest_of, standard_deviation_of "
            "and count_rows. Each one can take for_each=.",
            "A chart never changes its table, and a table step never changes a chart you already "
            "made. So one cleaned table can feed many charts.",
            "table.add_rows_from(other) puts the rows of another table under this one. "
            "turn_rows_into_columns does the opposite of turn_columns_into_rows.",
            "Lesson 13 teaches numbers, and lesson 14 the statistical tests: is a difference "
            "real, or could it be chance?",
            "Every summary skips empty cells and says how many it skipped. summary_of can also show "
            '"variance", "standard error", "confidence interval", "range" and "interquartile range".',
        ],
        task=[
            'Make the fruit totals for_each="shop" instead. Which shop sold more?',
            'Work out the average temperature for_each="city" in the weather table and draw it as bars.',
            "Is there a link between height and marks? Use correlation_between to find out.",
            'Make a cross table of the average mark for_each="subject" across="class". Which class is '
            "best in Science?",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("fruit"))
                totals = table.total_of("count", for_each="shop")
                print(totals)"""),
             "The Market stall sold more: 457 pieces of fruit, against 354 at the Corner shop."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                averages = table.average_of("temperature", for_each="city")
                chart = averages.chart()
                chart.bars(x="city", y="average of temperature")
                chart.numbers_on_bars(decimals=1)
                chart.save("lesson12_temperature.png")"""),
             "Frostby 1.0, Rainbridge 10.1 and Sunnyvale 17.0 degrees. The summary table's column is "
             'called "average of temperature".'),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.correlation_between("height", "marks")"""),
             "No: r = 0.03, very close to 0, and p = 0.83 says it could easily be chance."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.average_of("marks", for_each="subject", across="class")"""),
             "Class A, with an average of 79.2 in Science, against 67.5 for B and 62.3 for C."),
        ],
        words=["table.average_of", "table.biggest_of", "table.correlation_between", "table.total_of",
               "table.chart", "sdata.side_by_side", "table.join_with", "table.turn_columns_into_rows",
               "table.add_rows_from", "table.summary_of", "table.percent_of", "table.count_rows",
               "table.count_different_values"],
    ),
    dict(
        title="Numbers: lists and grids of numbers",
        learn=[
            "how to make a list of numbers, and a grid of them",
            "two ways to do maths: symbols make new numbers, words change the numbers you have",
            "how to keep some numbers, sum them up, and draw them",
        ],
        parts=[
            ("Numbers are for when you have just numbers, not a table with named columns.", [
                ("import simplydata as sdata", "Load simplydata."),
                ("nums = sdata.numbers([12, 15, 9, 20, 14])", "Make a list of five numbers and call it nums."),
                ("print(nums)", "Look at them. The first line says how many there are."),
                ("bigger = nums * 2", "Maths with symbols makes NEW numbers: every number times 2. "
                                      "nums itself does not change."),
                ("print(bigger)", "24, 30, 18, 40 and 28."),
                ("nums.add(1)", "Maths with words CHANGES the numbers you have: add 1 to each one."),
                ("nums.undo()", "Like a table, numbers can take back every word."),
                ("nums.keep_numbers(above=10)", "Keep only the numbers above 10. The same conditions "
                                                "as keep_rows: above, below, between ..."),
                ("nums.average()", "The average of what is left, said in a sentence."),
                ("nums.standard_deviation()", "How spread out they are (divide by n - 1, as in pandas "
                                              "and R)."),
                ("nums.steps()", "List the steps you did."),
            ]),
            ("Counting, random numbers and a chart.", [
                ("import simplydata as sdata", "Load simplydata."),
                ("months = sdata.numbers(1, to=12)", "Count from 1 to 12. The end is included."),
                ("temps = sdata.numbers([4, 6, 9, 13, 17, 20, 22, 21, 17, 12, 8, 5])", "A temperature for "
                                                                                    "each month."),
                ("chart = sdata.chart()", "Start an empty chart: the numbers go straight into the chart "
                                       "type."),
                ("chart.lines(x=months, y=temps, dots=True)", "Draw the temperatures as a line, month by "
                                                              "month."),
                ('chart.save("lesson13_line.png")', "Save it."),
                ("heights = sdata.random_numbers(how_many=200, average=150, standard_deviation=8, seed=1)",
                 "200 made-up heights that bunch around 150 in a bell shape (a normal distribution). "
                 "seed=1 gives the same numbers every time, so others can repeat your work."),
                ("heights.median()", "The middle height."),
                ("heights.percentile(90)", "The height that 90 % of them are below."),
                ("chart = heights.chart()", "Start a chart from the numbers. Their column is called "
                                            "value."),
                ('chart.histogram(x="value", bins=12)', "Count how many heights fall in each range."),
                ('chart.save("lesson13_heights.png")', "Save it. Can you see the bell shape?"),
            ]),
            ("A grid holds numbers in rows and columns. Positions count from 1.", [
                ("import simplydata as sdata", "Load simplydata."),
                ("grid = sdata.number_grid([[3, 5, 2], [8, 6, 4]])", "A grid with 2 rows and 3 columns: one "
                                                                  "list for each row."),
                ("print(grid.number_at(row=1, column=2))", "The number in row 1, column 2: 5."),
                ('grid.total(for_each="row")', "The total of each row: 10 and 18."),
                ('grid.average(for_each="column")', "The average of each column."),
                ("grid.swap_rows_and_columns()", "Rows become columns (in maths: the transpose)."),
                ("print(grid)", "Now 3 rows and 2 columns."),
                ("chart = grid.chart()", "Start a chart from the grid."),
                ("chart.heatmap(numbers=True)", "One coloured square for each number, with the number "
                                                "written in."),
                ('chart.save("lesson13_grid.png")', "Save it."),
            ]),
        ],
        notes=[
            "The words follow the table words, with numbers in place of rows: keep_first_numbers, "
            "remove_empty_numbers, fill_empty_numbers, show_first_numbers.",
            "Summaries have no _of, because there is no column to name: nums.average(), not "
            'table.average_of("marks").',
            "A sum with no answer, such as dividing by 0 or the square root of a negative number, "
            "gives an empty value and a note, never a crash.",
            "nums.table() turns numbers into a table, and nums.as_numpy() gives a plain NumPy array "
            "if another Python tool needs one.",
        ],
        task=[
            "Make the numbers 1 to 10, then square them in two ways: with ** 2 and with power(2).",
            "Make 6 dice throws with sdata.random_numbers(how_many=6, between=[1, 6], whole_numbers=True, "
            "seed=1). What is their total?",
            "Make a grid of 3 rows and 4 columns filled with 0: sdata.number_grid(rows=3, columns=4, "
            "filled_with=0).",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                nums = sdata.numbers(1, to=10)
                squares = nums ** 2
                print(squares)
                nums.power(2)
                nums.show()"""),
             "Both give 1, 4, 9 ... 100. ** 2 makes new numbers and leaves nums alone; power(2) "
             "changes nums itself (and undo() can take it back)."),
            (_sx("""\
                import simplydata as sdata
                dice = sdata.random_numbers(how_many=6, between=[1, 6], whole_numbers=True, seed=1)
                dice.show()
                dice.total()"""),
             "With seed=1 the throws are 3, 4, 5, 6, 1 and 1, which add up to 20."),
            (_sx("""\
                import simplydata as sdata
                grid = sdata.number_grid(rows=3, columns=4, filled_with=0)
                print(grid)"""),
             "Three rows of four zeros: a grid to fill in later."),
        ],
        words=["sdata.numbers", "sdata.random_numbers", "sdata.number_grid", "nums.add", "nums.undo",
               "nums.keep_numbers", "nums.average", "nums.standard_deviation", "nums.median",
               "nums.percentile", "nums.chart", "grid.number_at", "grid.swap_rows_and_columns", "nums.steps"],
    ),
    dict(
        title="Statistical tests: real difference, or chance?",
        learn=[
            "what a statistical test asks, and what a p-value means",
            "how to compare the averages of groups, and which groups differ",
            "how to compare counts, and how to fit a line or a formula",
        ],
        parts=[
            ("Groups almost never have exactly the same average. A test asks: is the difference "
             "bigger than chance alone would often give?", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.compare_averages("marks", for_each="class")',
                 "Do the three classes really differ in their average mark? The answer comes in "
                 "words. The p-value is the chance of a difference this big if the classes were "
                 "really the same: below 0.05 we call it unlikely to be chance. With three or more "
                 "groups this test is also called Welch's ANOVA; with two groups, a t-test."),
                ('table.compare_each_pair("marks", for_each="class")',
                 "Which classes differ? Each pair gets its own t-test (Welch's). Testing many pairs makes a "
                 "false alarm more likely, so the p-values are adjusted (Holm's method)."),
                ('table.compare_averages("marks", for_each="subject", based_on="ranks")',
                 "based_on=\"ranks\" puts the marks in order and compares the places instead "
                 "(a Kruskal-Wallis test, or Mann-Whitney for two groups). It is safer when groups "
                 "are small and the values are not bell-shaped."),
            ]),
            ("Counts can be compared too: does how often something happens differ between groups?", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.add_group_column("result", from_column="marks", cut_at=[60], '
                 'names=["below 60", "60 or more"])', "A new column that says if each mark is below "
                                                      "60 or not."),
                ('table.compare_counts("result", for_each="class")',
                 "Do the classes differ in how many marks are 60 or more? This counts the rows for "
                 "each class and result and uses the chi-square test. Cramer's V says how strong "
                 "the link is, from 0 (none) to 1."),
            ]),
            ("A fit finds the line or formula that follows the data best, and says how sure it is.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.fit_straight_line(x="hours", y="marks")',
                 "How much do the marks go up for each extra hour of study? This is linear regression "
                 "by least squares. R squared says how much of the differences in marks the line "
                 "explains."),
                ('growth = sdata.table(sdata.example("growth"))', "A second table, with the growth data."),
                ('growth.keep_rows("country", equal_to="Applia")', "Keep one country."),
                ('growth.fit_formula("population = a * exp(b * (year - 2000))")',
                 "Fit a formula you write yourself. Names that are columns (population, year) are "
                 "data; the others (a and b) are found by simplydata. Here b is the growth rate "
                 "each year. This is nonlinear least squares."),
            ]),
            ("Before a test, look at the shape of the values and at values far from the rest. After "
             "many tests, adjust the p-values.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Make a table of the marks."),
                ('table.compare_shape_to("marks", "normal", for_each="class")',
                 "Do the marks of each class look bell-shaped (normal)? The Shapiro-Wilk test answers for "
                 "each class. Tests on averages work best with bell-shaped values, or with many values."),
                ('table.show_rows_with_outliers("marks", for_each="class")',
                 "The rows whose mark is far from the rest of its class, by Tukey's rule (the one box "
                 "plots use): here one mark of 33 in class B. The table does not change."),
                ('pairs = table.correlation_between("hours", "marks", "height", show="each pair")',
                 "Test three pairs of columns at once: a table with one row per pair and its p-value."),
                ('pairs.add_adjusted_p_value_column(method="holm")',
                 "With many tests, one of them looks real by chance more often. Holm's adjustment keeps "
                 "the chance of even one false alarm at 5 %."),
                ("pairs.show()", "The p-values and the adjusted ones, side by side."),
            ]),
        ],
        notes=[
            "The idea behind every test: imagine there is no real difference (this is called the "
            "null hypothesis). The p-value says how surprising your data would be then. A small "
            "p-value means: hard to explain by chance alone.",
            "A p-value does not say how big or important the difference is: the effect size does "
            "that (omega-squared, Cohen's d, Cramer's V ...). And a large p-value does not show that "
            "there is no difference. It only means these data cannot tell.",
            "A 95 % confidence interval is the range the true value is very likely in. A narrow one "
            "means the data pin the value down well; a wide one means you need more data.",
            "Each test gives back a one-row table with all its numbers, so you can save it: "
            'result = table.compare_averages(...), then result.save("test.csv").',
            "Also: compare_before_after for two measurements of the same thing (a paired t-test), "
            "and compare_average_to for one average against a number you choose.",
            "remove_rows_with_outliers removes such rows. Say in your report which rule removed how many "
            "rows, and never remove a value only because it spoils the result.",
            'method="benjamini-hochberg" adjusts for many tests more gently: it controls the share of '
            "false alarms among your findings (the false discovery rate).",
        ],
        task=[
            'Compare the average marks for_each="subject". Could the differences be chance?',
            'Is the average temperature different between the cities of sdata.example("weather")?',
            'Fit a straight line of "forest" against "year" in sdata.example("growth") for one country.',
            'Remove the rows with outliers in "marks" for_each="class", then compare the average marks '
            'for_each="class" again. Did the answer change?',
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.compare_averages("marks", for_each="subject")"""),
             "Yes, they could: p = 0.26 is well above 0.05. That does not prove the subjects are "
             "the same; these data simply cannot tell."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.compare_averages("temperature", for_each="city")
                table.compare_each_pair("temperature", for_each="city")"""),
             "Yes: p = 0.0002, very unlikely to be chance. compare_each_pair then shows which "
             "cities differ from each other."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("growth"))
                table.keep_rows("country", equal_to="Applia")
                table.fit_straight_line(x="year", y="forest")"""),
             "In Applia the forest shrinks by about 0.57 percentage points a year (95 % CI 0.48 to "
             "0.67), and the line explains 94 % of the differences."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("marks"))
                table.remove_rows_with_outliers("marks", for_each="class")
                table.compare_averages("marks", for_each="class")"""),
             "Only one mark (33 in class B) is removed. Class B goes up from 64.3 to 66.1, but p stays 0.028: "
             "the answer does not change. Say in a report that you checked, and how."),
        ],
        words=["table.compare_averages", "table.compare_each_pair", "table.compare_counts",
               "table.fit_straight_line", "table.fit_formula", "table.add_group_column",
               "table.compare_shape_to", "table.show_rows_with_outliers", "table.add_adjusted_p_value_column",
               "table.remove_rows_with_outliers"],
    ),
    dict(
        title="A whole analysis: from a question to a figure",
        learn=[
            "how to put all the lessons together: open, look, clean, sum up, test, draw and save",
            "how to make a figure with panels A and B, ready for a report or a paper",
            "how steps() writes down what you did, in plain words",
            "how to hand your table to pandas or Polars, and back",
        ],
        parts=[
            ("A whole analysis with one question: do children who study longer get better marks, "
             "and do the classes differ? One table feeds two charts, and the two charts become "
             "one figure.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("marks"))', "Open the data. For your own file: "
                                                          'sdata.table("results.xlsx").'),
                ("table.describe_columns()", "Look: what is in each column, and are any cells empty?"),
                ("table.remove_rows_with_empty_cells()", "Clean: remove the rows with an empty cell. The "
                                                         "practice table has none, so nothing is removed."),
                ('table.average_of("marks", for_each="class")', "Sum up: the average mark of each class."),
                ('table.compare_averages("marks", for_each="class")', "Test: could the differences "
                                                                     "between the classes be chance?"),
                ("a = table.chart()", "Chart A, from the cleaned table."),
                ('a.boxes(x="class", y="marks")', "The marks of each class as boxes."),
                ('a.all_dots(x="class", y="marks")', "Every mark as a dot, on top of the boxes."),
                ('a.label_y("Mark (out of 100)")', "Name its y-axis."),
                ("b = table.chart()", "Chart B, from the same table."),
                ('b.dots(x="hours", y="marks", colour="class")', "Hours of study against marks, "
                                                                 "one colour per class."),
                ("b.fit_line(confidence_band=True)", "A straight line for each class, with its 95 % "
                                                     "confidence band."),
                ('b.label_x("Hours of study")', "Name its x-axis."),
                ("figure = sdata.grid(a, b, per_row=2, letters=True)", "Both charts on one page, with A "
                                                                    "and B at the top left."),
                ('figure.title_top("Marks, classes and hours of study")', "A title for the whole page."),
                ('figure.style("paper")', "A quiet look that suits a paper."),
                ('figure.save("lesson15_figure.pdf")', "Save the figure as a PDF: sharp at any size."),
                ('table.save("lesson15_clean.xlsx")', "Save the cleaned table that the figure was "
                                                      "drawn from."),
                ("table.steps()", "Every table step in plain words, in order: a good start for the "
                                  "methods part of your report."),
            ]),
            ("Two questions about two tables give two charts. sdata.grid puts them on one page.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('weather = sdata.table(sdata.example("weather"))', "Table 1: the weather."),
                ('weather.keep_rows("city", equal_to="Rainbridge")', "Keep one city."),
                ("rain = weather.chart()", "A chart of it, called rain."),
                ('rain.bars(x="month", y="rain")', "Its rain through the year."),
                ('rain.line_at_y(100, label="100 mm")', "A line at 100 mm."),
                ('growth = sdata.table(sdata.example("growth"))', "Table 2: four countries."),
                ("forest = growth.chart()", "A chart of it, called forest."),
                ('forest.lines(x="year", y="forest", colour="country")', "Their forest over the years."),
                ('forest.label_y("Forest (% of the land)")', "Name the y-axis."),
                ("page = sdata.grid(rain, forest, per_row=2, letters=True)", "Both charts on one page, "
                                                                          "with letters."),
                ('page.save("lesson15_two_tables.png")', "Save the page."),
            ]),
            ("A table goes to pandas or Polars and back whenever another Python tool needs it.", [
                ("import simplydata as sdata", "Load simplydata."),
                ('table = sdata.table(sdata.example("weather"))', "A table of the weather."),
                ('table.keep_rows("city", equal_to="Rainbridge")', "Keep one city."),
                ("df = table.as_pandas()", "A plain pandas DataFrame (table.as_polars() gives a "
                                           "Polars one)."),
                ('print(df["rain"].max())', "Any pandas code works on it."),
                ("again = sdata.table(df)", "And back: sdata.table takes a pandas or Polars table you "
                                         "already have."),
                ("chart = sdata.chart(df)", "sdata.chart takes one too."),
                ('chart.bars(x="month", y="rain")', "Draw it."),
                ('chart.save("lesson15_rain.png")', "Save the chart."),
            ]),
        ],
        notes=[
            "Every analysis follows the same plan: open, look, clean, sum up, test, draw, words, "
            "save. Lessons 1 to 14 taught each part; this lesson puts them together.",
            "The table and the charts are separate objects, as in pandas and matplotlib: one cleaned "
            "table can feed many charts, and each chart keeps the rows the table had when the chart "
            "was made.",
            "table.save saves the table, and chart.save or page.save the picture. Save both, so "
            "anyone can check the figure against its data.",
            "table.steps() lists the table steps. Together with the chart lines of your program, it "
            "says everything you did.",
            "simplydata words never need as_pandas(); it is only for other tools.",
        ],
        task=[
            'Do a whole analysis of sdata.example("weather"): look at it, compare the average rain '
            'for_each="city", draw the rain of each city as boxes, and save the chart and the table.',
            "Make a second chart of the temperature, then put the two charts together with "
            "sdata.side_by_side(..., letters=True).",
            "Read the steps() of your table out loud. Would someone else understand what you did?",
        ],
        solutions=[
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.describe_columns()
                table.remove_columns("wind_direction", "wind_speed")
                table.average_of("rain", for_each="city")
                table.compare_averages("rain", for_each="city")
                chart = table.chart()
                chart.boxes(x="city", y="rain")
                chart.all_dots(x="city", y="rain")
                chart.label_y("Rain (mm)")
                chart.save("lesson15_rain_boxes.png")
                table.save("lesson15_weather.xlsx")"""),
             "Rainbridge is much wetter (109 mm a month on average) than Frostby (42) and Sunnyvale "
             "(26); p < 0.0001, so this is very unlikely to be chance."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                rain = table.chart()
                rain.boxes(x="city", y="rain")
                rain.label_y("Rain (mm)")
                warmth = table.chart()
                warmth.lines(x="month", y="temperature", colour="city")
                warmth.label_y("Temperature (°C)")
                page = sdata.side_by_side(rain, warmth, letters=True)
                page.save("lesson15_weather_page.png")"""),
             "Two charts of one table, side by side, with A and B so a report can point to each."),
            (_sx("""\
                import simplydata as sdata
                table = sdata.table(sdata.example("weather"))
                table.remove_columns("wind_direction", "wind_speed")
                table.keep_rows("month_number", between=[6, 8])
                table.sort_by("rain", biggest_first=True)
                table.steps()"""),
             "steps() lists each change in plain words, in order; those lines are a first draft of "
             "the methods part of a report."),
        ],
        words=["sdata.table", "table.describe_columns", "table.remove_rows_with_empty_cells",
               "table.compare_averages", "table.chart", "boxes", "all_dots", "dots", "fit_line",
               "sdata.grid", "style", "table.save", "table.steps", "table.as_pandas", "sdata.chart"],
    ),
]

def _wrap(text: str, indent: str = "", width: int = WIDTH) -> str:
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def lesson_code(number: int) -> list[str]:
    """The code of each part of a lesson, ready to run (used by the tests)."""
    return ["\n".join(code for code, _ in steps) for _, steps in LESSONS[number - 1]["parts"]]


def _list_lessons() -> None:
    print(f"Learn simplydata in one day: {len(LESSONS)} lessons of about 30 minutes.")
    print("Lessons 1 to 8 teach charts, lessons 9 to 12 teach tables, lesson 13 teaches numbers,")
    print("lesson 14 teaches statistical tests and lesson 15 puts it all together, from a question")
    print("to a figure.\n")
    for i, les in enumerate(LESSONS, start=1):
        print(f"  sdata.lesson({i}){' ' * (5 - len(str(i)))}{les['title']}")
    print("\nStart with sdata.lesson(1). Each lesson ends with a small task to try yourself.")


def lesson_solutions(number: int) -> list[tuple[str, str]]:
    """The worked solution of each task of a lesson: [(code, explanation), ...]."""
    return list(LESSONS[number - 1]["solutions"])


def lesson(number=None, solutions=False) -> None:
    """A lesson: an example to run, what each line does, and a task. sdata.lesson(1)

    sdata.lesson() on its own lists all the lessons. solutions=True also shows a worked
    solution for each task: sdata.lesson(3, solutions=True).
    """
    if number is None:
        _list_lessons()
        return None
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= len(LESSONS):
        raise SimplydataError(
            f"There are lessons 1 to {len(LESSONS)}, so sdata.lesson({number!r}) does not exist.\n"
            "Start with sdata.lesson(1), or type sdata.lesson() to see them all."
        )
    les = LESSONS[number - 1]
    title = f"Lesson {number} of {len(LESSONS)}: {les['title']}"
    print(title)
    print("=" * len(title))
    print("\nIn this lesson you will learn")
    for item in les["learn"]:
        print(f"  - {item}")
    for i, (intro, steps) in enumerate(les["parts"], start=1):
        print()
        heading = "Example" if len(les["parts"]) == 1 else f"Example {i}"
        print(heading)
        print("-" * len(heading))
        print(_wrap(intro))
        print()
        for code, _ in steps:
            print(f"    {code}")
        print("\nWhat each line does")
        for code, explain in steps:
            if not explain:
                continue
            print(f"    {code}")
            print(_wrap(explain, "        "))
    if les.get("notes"):
        print("\nGood to know")
        for note in les["notes"]:
            print(textwrap.fill(note, WIDTH, initial_indent="  - ", subsequent_indent="    "))
    print("\nTry it yourself")
    for i, task in enumerate(les["task"], start=1):
        print(textwrap.fill(task, WIDTH, initial_indent=f"  {i}. ", subsequent_indent="     "))
    if solutions:
        print("\nSolutions")
        for i, (code, explain) in enumerate(les["solutions"], start=1):
            print()
            print(textwrap.fill(explain, WIDTH, initial_indent=f"  {i}. ", subsequent_indent="     "))
            print(textwrap.indent(code, "        "))
    else:
        print(f"\nStuck? sdata.lesson({number}, solutions=True) shows a worked solution for each task.")
    print()
    if number < len(LESSONS):
        print(f"Next: sdata.lesson({number + 1})  {LESSONS[number]['title']}")
    else:
        print("You have finished all the lessons. Well done!")
    return None
