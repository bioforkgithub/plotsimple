"""Learning plotsimple: ps.words(), ps.help("bars") and ps.lesson(1) ... ps.lesson(15).

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

from ._errors import ISSUES_URL, PlotsimpleError, suggest
from .story import DRAW_WORDS

WIDTH = 78

# ---------------------------------------------------------------- the grouping
# "ps.name" is a function you call on ps; "name=" is an option used inside chart steps;
# a plain name is a step on a chart (chart.name(...)); "table.name" is a step on a table
# (table.name(...)); "nums.name" is a step on numbers and "grid.name" a step on a grid of
# numbers. "story.name" is a word a story has in its own way (story.draw_bars(...)); the
# other table and chart words work on a story too and are explained once, as table and chart
# words. Other spellings of a word (bar, scatter, color ...) are found by themselves and
# shown next to it. Old names (kept for one release) are never listed here.
GROUPS = [
    ("Story: a whole analysis in one object", ["ps.start"] + [f"story.draw_{kind}" for kind in DRAW_WORDS] + [
        "story.new_chart", "story.put_charts_together", "story.show_chart", "story.save_chart",
        "story.save_table", "story.save", "story.undo", "story.steps", "story.start_again"]),
    ("Start, show and save", ["ps.chart", "ps.example", "show", "save"]),
    ("Chart types", ["bars", "lines", "dots", "area", "histogram", "boxes", "violins",
                     "heatmap", "pie", "polar", "contour", "surface", "dots_3d"]),
    ("Statistics", ["show=", "error=", "error_bars", "fit_line", "fit_curve", "smooth_line",
                    "density", "count_bars", "all_dots", "side_histograms", "ps.pairs"]),
    ("Lines and shading", ["line_at_x", "line_at_y", "shade_between_x", "shade_between_y"]),
    ("Words on the chart", ["title_top", "caption_bottom", "label_x", "label_y", "note",
                            "numbers_on_bars", "font_size"]),
    ("Axes", ["range_x", "range_y", "log_scale_x", "log_scale_y", "ticks_x", "ticks_y",
              "rotate_labels_x", "rotate_labels_y", "second_y", "flip_x", "flip_y"]),
    ("Colours and look", ["colour", "colours", "style", "size", "show_grid", "hide_grid",
                          "legend_top", "legend_bottom", "legend_left", "legend_right",
                          "hide_legend", "show_legend", "label_colour_bar"]),
    ("Several charts", ["one_chart_per", "ps.side_by_side", "ps.one_above_other", "ps.grid",
                        "zoom_box", "animate"]),
    # Table words: "table.name" is a step on a table (table.name(...)).
    ("Tables: start and look", ["ps.table", "table.show_first_rows", "table.show_last_rows",
                                "table.column_names", "table.describe_columns"]),
    ("Choose columns", ["table.keep_columns", "table.remove_columns", "table.rename_column"]),
    ("Choose rows", ["table.keep_rows", "table.remove_rows", "table.keep_first_rows",
                     "table.keep_last_rows", "table.keep_random_rows", "table.remove_duplicate_rows"]),
    ("Empty cells", ["table.count_empty_cells", "table.remove_rows_with_empty_cells",
                     "table.fill_empty_cells"]),
    ("Fix columns", ["table.make_numbers", "table.make_text", "table.make_dates",
                     "table.make_lowercase", "table.make_uppercase", "table.remove_extra_spaces",
                     "table.replace_values"]),
    ("Sort and new columns", ["table.sort_by", "table.add_column", "table.add_group_column",
                              "table.add_rank_column"]),
    ("Summaries", ["table.count_rows", "table.count_each", "table.average_of", "table.median_of",
                   "table.total_of", "table.smallest_of", "table.biggest_of",
                   "table.standard_deviation_of", "table.correlation_between"]),
    ("Statistical tests", ["table.compare_averages", "table.compare_each_pair", "table.compare_before_after",
                           "table.compare_average_to", "table.compare_counts", "table.fit_straight_line",
                           "table.fit_formula"]),
    ("Combine and reshape", ["table.join_with", "table.add_rows_from", "table.turn_columns_into_rows",
                             "table.turn_rows_into_columns"]),
    ("Undo and finish", ["table.undo", "table.steps", "table.start_again", "table.save",
                         "table.chart", "table.as_pandas", "table.as_polars"]),
    # Number words: "nums.name" is a step on numbers, "grid.name" a step on a grid of numbers.
    ("Numbers: make and look", ["ps.numbers", "ps.random_numbers", "nums.show", "nums.show_first_numbers",
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
    ("Grids of numbers", ["ps.number_grid", "nums.number_grid", "grid.count_rows", "grid.count_columns",
                          "grid.shape", "grid.show_first_rows", "grid.show_last_rows", "grid.number_at",
                          "grid.row", "grid.column", "grid.numbers", "grid.swap_rows_and_columns"]),
    ("Numbers: undo and finish", ["nums.undo", "nums.steps", "nums.start_again", "nums.table", "nums.chart",
                                  "nums.save", "nums.as_numpy"]),
    ("Help", ["ps.words", "ps.help", "ps.lesson"]),
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
        'ps.help("show=").'
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
    lines = ["import plotsimple as ps", f'{table} = ps.example("{table}")',
             f"chart = ps.chart({table})"]
    lines += textwrap.dedent(body).strip().splitlines()
    lines.append(f'chart.save("{file}.png")')
    return "\n".join(lines)


def _tx(name: str, body: str) -> str:
    """A runnable table example: start a table from a practice table, then the steps."""
    lines = ["import plotsimple as ps", f'table = ps.table(ps.example("{name}"))']
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


def _nx(body: str, values: str = "[12, 15, 9, 20, 14]") -> str:
    """A runnable numbers example: make some numbers, then the steps."""
    lines = ["import plotsimple as ps", f"nums = ps.numbers({values})"]
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


def _gx(body: str) -> str:
    """A runnable grid example: make a grid of numbers, then the steps."""
    lines = ["import plotsimple as ps", "grid = ps.number_grid([[3, 5, 2], [8, 6, 4]])"]
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


HELP_EXAMPLES = {
    # start, show, save
    "ps.chart": _ex("marks", 'chart.dots(x="hours", y="marks")', "my_chart"),
    "ps.example": textwrap.dedent("""\
        import plotsimple as ps
        ps.example()
        marks = ps.example("marks")
        print(marks)"""),
    "show": _ex("marks", 'chart.bars(x="name", y="marks")\nchart.show()', "marks"),
    "save": textwrap.dedent("""\
        import plotsimple as ps
        marks = ps.example("marks")
        chart = ps.chart(marks)
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
    "ps.pairs": textwrap.dedent("""\
        import plotsimple as ps
        marks = ps.example("marks")
        chart = ps.pairs(marks, colour="class")
        chart.title_top("Every number column against every other")
        chart.save("pairs.png")"""),
    # lines and shading
    "line_at_x": _ex("growth", 'chart.lines(x="year", y="population", colour="country")\nchart.line_at_x(2010, label="2010")', "line_at_x"),
    "line_at_y": _ex("marks", 'chart.bars(x="name", y="marks")\nchart.line_at_y(50, label="pass mark")\nchart.rotate_labels_x()', "line_at_y"),
    "shade_between_x": _ex("weather", 'chart.lines(x="month_number", y="temperature", colour="city")\nchart.shade_between_x(6, 8)', "shade_between_x"),
    "shade_between_y": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.shade_between_y(40, 60)', "shade_between_y"),
    # words on the chart
    "title_top": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.title_top("Marks in each class")', "title_top"),
    "caption_bottom": _ex("marks", 'chart.bars(x="class", y="marks")\nchart.caption_bottom("Made-up data from ps.example")', "caption_bottom"),
    "label_x": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.label_x("Hours of study")', "label_x"),
    "label_y": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.label_y("Marks (out of 100)")', "label_y"),
    "note": _ex("growth", 'chart.lines(x="year", y="population", colour="country")\nchart.note("Cherria grows fastest", x=2020, y=12, arrow=True)', "note"),
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
    "label_colour_bar": _ex("island", 'chart.contour(x="east", y="north", z="height")\n'
                            'chart.label_colour_bar("Height (m)")', "label_colour_bar"),
    # several charts
    "one_chart_per": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.one_chart_per("class", per_row=3)',
                         "one_chart_per"),
    "ps.side_by_side": textwrap.dedent("""\
        import plotsimple as ps
        marks = ps.example("marks")
        a = ps.chart(marks).bars(x="class", y="marks").title_top("Average")
        b = ps.chart(marks).boxes(x="class", y="marks").title_top("Spread")
        page = ps.side_by_side(a, b, same_y=True)
        page.save("side_by_side.png")"""),
    "ps.one_above_other": textwrap.dedent("""\
        import plotsimple as ps
        weather = ps.example("weather")
        a = ps.chart(weather).lines(x="month", y="temperature", colour="city")
        b = ps.chart(weather).bars(x="month", y="rain", colour="city")
        page = ps.one_above_other(a, b)
        page.save("one_above_other.png")"""),
    "ps.grid": textwrap.dedent("""\
        import plotsimple as ps
        marks = ps.example("marks")
        a = ps.chart(marks).bars(x="class", y="marks")
        b = ps.chart(marks).histogram(x="marks")
        c = ps.chart(marks).dots(x="hours", y="marks")
        d = ps.chart(marks).boxes(x="subject", y="marks")
        page = ps.grid(a, b, c, d, per_row=2)
        page.title_top("Four ways to look at the marks")
        page.save("grid.png")"""),
    "zoom_box": _ex("marks", 'chart.dots(x="hours", y="marks")\n'
                    'chart.zoom_box(x_between=[2, 4], y_between=[40, 60], where="top left")', "zoom_box"),
    "animate": textwrap.dedent("""\
        import plotsimple as ps
        growth = ps.example("growth")
        chart = ps.chart(growth)
        chart.bars(x="country", y="population")
        chart.range_y(0, 50)
        chart.animate(over="year")
        chart.save("growth.gif")"""),
    # tables: start and look
    "ps.table": _tx("marks", "table.show_first_rows()"),
    "table.show_first_rows": _tx("marks", "table.show_first_rows(10)"),
    "table.show_last_rows": _tx("marks", "table.show_last_rows(3)"),
    "table.column_names": _tx("marks", "print(table.column_names())"),
    "table.describe_columns": _tx("marks", "table.describe_columns()"),
    # choose columns
    "table.keep_columns": _tx("marks", 'table.keep_columns("name", "marks")\ntable.show_first_rows()'),
    "table.remove_columns": _tx("marks", 'table.remove_columns("height")\nprint(table.column_names())'),
    "table.rename_column": _tx("marks", 'table.rename_column("marks", to="score")\ntable.show_first_rows()'),
    # choose rows
    "table.keep_rows": _tx("marks", 'table.keep_rows("marks", above=50)\n'
                                    'table.keep_rows("subject", one_of=["Maths", "Science"])\ntable.show_first_rows()'),
    "table.remove_rows": _tx("marks", 'table.remove_rows("subject", equal_to="English")\ntable.show_first_rows()'),
    "table.keep_first_rows": _tx("marks", "table.keep_first_rows(10)\nprint(table)"),
    "table.keep_last_rows": _tx("marks", "table.keep_last_rows(10)\nprint(table)"),
    "table.keep_random_rows": _tx("marks", "table.keep_random_rows(10, seed=1)\nprint(table)"),
    "table.remove_duplicate_rows": _tx("marks", 'table.remove_duplicate_rows(columns=["name"])\nprint(table)'),
    # empty cells (the practice tables have none, so the examples make some first)
    "table.count_empty_cells": _tx("marks", 'table.replace_values("marks", [61, 73], with_value=None)\n'
                                            "table.count_empty_cells()"),
    "table.remove_rows_with_empty_cells": _tx("marks", 'table.replace_values("marks", [61, 73], with_value=None)\n'
                                                       'table.remove_rows_with_empty_cells("marks")\nprint(table)'),
    "table.fill_empty_cells": _tx("marks", 'table.replace_values("marks", [61, 73], with_value=None)\n'
                                           'table.fill_empty_cells("marks", with_the="average")\n'
                                           "table.show_first_rows()"),
    # fix columns
    "table.make_numbers": _tx("marks", 'table.make_numbers("marks")\ntable.describe_columns()'),
    "table.make_text": _tx("weather", 'table.make_text("month_number")\ntable.describe_columns()'),
    "table.make_dates": _tx("growth", 'table.make_dates("year")\ntable.show_first_rows()'),
    "table.make_lowercase": _tx("marks", 'table.make_lowercase("name")\ntable.show_first_rows()'),
    "table.make_uppercase": _tx("marks", 'table.make_uppercase("name")\ntable.show_first_rows()'),
    "table.remove_extra_spaces": _tx("marks", 'table.remove_extra_spaces("name")\ntable.show_first_rows()'),
    "table.replace_values": _tx("marks", 'table.replace_values("subject", "Maths", with_value="Math")\n'
                                         'table.count_each("subject")'),
    # sort and new columns
    "table.sort_by": _tx("marks", 'table.sort_by("marks", biggest_first=True)\ntable.show_first_rows()'),
    "table.add_column": _tx("marks", 'table.add_column("study_and_marks", by_adding=["hours", "marks"])\n'
                                     'table.add_column("marks_per_hour", "marks / hours")\ntable.show_first_rows()'),
    "table.add_group_column": _tx("marks", 'table.add_group_column("grade", from_column="marks", '
                                           'cut_at=[40, 60, 80], names=["D", "C", "B", "A"])\n'
                                           'table.count_each("grade")'),
    "table.add_rank_column": _tx("marks", 'table.add_rank_column("place", from_column="marks")\n'
                                          'table.sort_by("place")\ntable.show_first_rows()'),
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
    "table.correlation_between": _tx("marks", 'table.correlation_between("hours", "marks")'),
    # statistical tests
    "table.compare_averages": _tx("marks", 'result = table.compare_averages("marks", between="class")\n'
                                           'table.compare_averages("marks", between="class", based_on="ranks")\n'
                                           'result.save("marks_test.csv")'),
    "table.compare_each_pair": _tx("marks", 'table.compare_each_pair("marks", between="class")'),
    "table.compare_before_after": _tx("marks", 'table.keep_columns("name", "subject", "marks")\n'
                                               'table.turn_rows_into_columns(names_from="subject", values_from="marks")\n'
                                               'table.compare_before_after("Maths", "Science")'),
    "table.compare_average_to": _tx("marks", 'table.compare_average_to("marks", 60)'),
    "table.compare_counts": _tx("marks", 'table.add_group_column("result", from_column="marks", cut_at=[60], '
                                         'names=["below 60", "60 or more"])\n'
                                         'table.compare_counts("result", between="class")'),
    "table.fit_straight_line": _tx("marks", 'table.fit_straight_line(x="hours", y="marks")'),
    "table.fit_formula": _tx("growth", 'table.keep_rows("country", equal_to="Applia")\n'
                                       'table.fit_formula("population = a * exp(b * (year - 2000))")'),
    # combine and reshape
    "table.join_with": _tx("marks", 'ages = {"name": ["Asha", "Ben"], "age": [12, 13]}\n'
                                    'table.join_with(ages, using="name", keep="all rows from this table")\n'
                                    "table.show_first_rows()"),
    "table.add_rows_from": _tx("marks", 'table.add_rows_from(ps.example("marks"))\nprint(table)'),
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
    "ps.numbers": textwrap.dedent("""\
        import plotsimple as ps
        nums = ps.numbers([3, 5, 2, 8])
        print(nums)
        print(ps.numbers(1, to=10))
        print(ps.numbers(0, to=1, how_many=5))"""),
    "ps.random_numbers": textwrap.dedent("""\
        import plotsimple as ps
        dice = ps.random_numbers(how_many=10, between=[1, 6], whole_numbers=True, seed=1)
        dice.show()
        heights = ps.random_numbers(how_many=100, average=150, standard_deviation=8, seed=1)
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
    "nums.fill_empty_numbers": _nx('nums.fill_empty_numbers(with_the="average")\nnums.show()', "[12, None, 9, 14]"),
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
    "ps.number_grid": textwrap.dedent("""\
        import plotsimple as ps
        grid = ps.number_grid([[1, 2, 3], [4, 5, 6]])
        print(grid)
        print(ps.number_grid(rows=2, columns=3, filled_with=0))"""),
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
    "ps.words": "import plotsimple as ps\nps.words()",
    "ps.help": 'import plotsimple as ps\nps.help("bars")\nps.help("show=")\nps.help("keep_rows")\nps.help("average")',
    "ps.lesson": "import plotsimple as ps\nps.lesson()\nps.lesson(1)",
}


def _as_story(code: str) -> str:
    """A chart help example told as a story: ps.start(example=...), draw_ words, save_chart."""
    lines = []
    for line in code.splitlines():
        start = re.fullmatch(r'\w+ = ps\.example\("(\w+)"\)', line)
        if start:
            lines.append(f'story = ps.start(example="{start.group(1)}")')
        elif not re.fullmatch(r"chart = ps\.chart\(\w+\)", line):
            line = re.sub(rf"\bchart\.({'|'.join(DRAW_WORDS)})\(", r"story.draw_\1(", line)
            line = line.replace("chart.save(", "story.save_chart(").replace("chart.show(", "story.show_chart(")
            lines.append(re.sub(r"(?<![\w.])chart\.", "story.", line))
    return "\n".join(lines)


def _sx(name: str, body: str) -> str:
    """A runnable story example: start a story from a practice table, then the steps."""
    lines = ["import plotsimple as ps", f'story = ps.start(example="{name}")']
    lines += textwrap.dedent(body).strip().splitlines()
    return "\n".join(lines)


HELP_EXAMPLES.update({
    "ps.start": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.draw_bars(x="class", y="marks")\n'
                             'story.title_top("Good marks")\nstory.save_chart("good_marks.png")'),
    **{f"story.draw_{kind}": _as_story(HELP_EXAMPLES[kind]) for kind in DRAW_WORDS},
    "story.new_chart": _sx("marks", 'story.draw_bars(x="class", y="marks")\nstory.save_chart("all_classes.png")\n'
                                    'story.new_chart()\nstory.keep_rows("class", equal_to="A")\n'
                                    'story.draw_dots(x="hours", y="marks")\nstory.save_chart("class_a.png")'),
    "story.put_charts_together": _sx("marks", 'story.draw_boxes(x="class", y="marks")\nstory.new_chart()\n'
                                              'story.draw_dots(x="hours", y="marks", colour="class")\n'
                                              "story.put_charts_together(per_row=2, letters=True)\n"
                                              'story.title_top("Marks")\nstory.save_chart("figure.png")'),
    "story.show_chart": _sx("marks", 'story.draw_bars(x="class", y="marks")\nstory.show_chart()'),
    "story.save_chart": _sx("marks", 'story.draw_bars(x="class", y="marks")\nstory.save_chart("results.png")\n'
                                     'story.save_chart("results.pdf")\n'
                                     'story.save_chart("results_sharp.png", quality="high")'),
    "story.save_table": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.save_table("clean.xlsx")\n'
                                     'story.save_table("clean.csv")'),
    "story.save": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.draw_bars(x="class", y="marks")\n'
                               'story.save("results.png")\nstory.save("clean.csv")'),
    "story.undo": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.draw_bars(x="class", y="marks")\n'
                               "story.undo()\nstory.undo()\nstory.steps()"),
    "story.steps": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.draw_bars(x="class", y="marks")\n'
                                'story.title_top("Good marks")\nstory.steps()'),
    "story.start_again": _sx("marks", 'story.keep_rows("marks", above=50)\nstory.draw_bars(x="class", y="marks")\n'
                                      "story.start_again()\nprint(story)"),
})


# ---------------------------------------------------------------- finding words
def _ps():
    import plotsimple  # late import: plotsimple imports this file

    return plotsimple


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
_PREFIX = {"table": "table", "numbers": "nums", "grid": "grid", "story": "story"}


def _story_class():
    from .story import Story

    return Story


def _classes():
    """(class, kind) for every object with steps."""
    return [(_chart_class(), "chart"), (_table_class(), "table"), (_numbers_class(), "numbers"),
            (_grid_class(), "grid"), (_story_class(), "story")]


def _kind(key: str) -> str:
    """What a key in GROUPS is: "option", "ps", "table", "numbers", "grid" or "chart"."""
    if key.endswith("="):
        return "option"
    if key.startswith("ps."):
        return "ps"
    for kind, prefix in _PREFIX.items():
        if key.startswith(prefix + "."):
            return kind
    return "chart"


def _word(key: str) -> str:
    """The word itself, without "ps.", "table." or "nums." in front: "ps.chart" -> "chart"."""
    kind = _kind(key)
    if kind == "ps":
        return key[3:]
    if kind in _PREFIX:
        return key[len(_PREFIX[kind]) + 1:]
    return key


def _display(key: str) -> str:
    """How a key is shown in ps.words() and the cheat sheet: table and number words without
    "table." or "nums." (their group says what they belong to)."""
    return _word(key) if _kind(key) in _PREFIX else key


def _function(key: str):
    """The real function behind a word in GROUPS (None for options such as "show=")."""
    kind = _kind(key)
    if kind == "option":
        return None
    if kind == "ps":
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
    return _PREFIX.get(_kind(key), "chart|ps|page")


def _find_call(text: str, prefix_word: str,
               prefixes: str = "chart|ps|page|table|nums|grid|story") -> str | None:
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


_TABLE = re.compile(r'ps\.(?:example\(|start\(example=)"(\w+)"')


def _example_table(key: str) -> str | None:
    """The example table a word's examples use: named in the docstring, else in the help example.

    A docstring names its table like this: chart.lines(...) (with ps.example("weather")).
    A help example names it with ps.example("weather") or ps.start(example="weather").
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


# ---------------------------------------------------------------- ps.words()
def words() -> None:
    """Print every plotsimple word with a short example. ps.words()

    For more about one word, type ps.help("bars").
    """
    aliases = _aliases()
    pad = max(len(_display(k)) for k in _all_keys()) + 2
    print("plotsimple words. Type ps.help(\"bars\") to learn more about any of them.")
    print('A story is one object for a whole analysis: story = ps.start(example="marks"), then')
    print("every table word and chart word on it; chart types get draw_: story.draw_bars(...).")
    print("Chart words are steps on a chart: chart = ps.chart(table), then chart.bars(...).")
    print("Table words are steps on a table: table = ps.table(...), then table.keep_rows(...).")
    print("Number words are steps on numbers: nums = ps.numbers([3, 5, 2]), then nums.average();")
    print("grid words are steps on a grid of numbers: grid = ps.number_grid([[1, 2], [3, 4]]).")
    print('The examples use the practice tables; ps.help("word") shows which one, ready to run.')
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


# ---------------------------------------------------------------- ps.help()
def _normalise(word: str) -> str:
    w = str(word).strip()
    for start in ("ps.", "chart.", "table.", "page.", "nums.", "grid.", "story.", "."):
        if w.startswith(start):
            w = w[len(start):]
    if w.endswith("()"):
        w = w[:-2]
    return w.strip()


def _chart_or_ps_key(w: str) -> str | None:
    """A chart word, ps word or option called w (already normalised), or None."""
    keys = _all_keys()
    if w in keys:
        return w
    if f"ps.{w}" in keys:
        return f"ps.{w}"
    if f"{w}=" in keys:
        return f"{w}="
    if w.endswith("=") and w[:-1] in keys:
        return w[:-1]
    for key, names in _aliases().items():
        if _kind(key) in ("chart", "ps") and w in names:
            return key
    if w == "Chart":
        return "ps.chart"
    if w == "Table":
        return "ps.table"
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
        if typed.startswith(("chart.", "page.", "ps.")):
            found = [_chart_or_ps_key(w)]
        else:
            found = [_chart_or_ps_key(w)] + [_object_key(w, kind) for kind in _PREFIX]
    return [k for k in dict.fromkeys(found) if k is not None]


def _old_names(word) -> list[tuple[str, str, str]]:
    """What an old name is called now: [(old, new, key of the new word), ...].

    Old words come from each object's _OLD_WORDS, old options from the renamed= of
    @friendly. Both still work for one release, but ps.words() never lists them.
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
        How to get help in plotsimple

          ps.help("bars")       explains one word: what it does, its options and an example
          ps.help("keep_rows")  works for table words too, and ps.help("average") for numbers
          ps.help("draw_bars")  works for story words, and says which chart word it draws
          ps.help("show=")      explains an option you use inside a step
          ps.words()            lists every word, with a short example of each
          ps.lesson()           lists the lessons: learn the whole of plotsimple in a day
          ps.example()          lists the made-up tables you can practise with

        You can type the word with or without "chart." or "table." in front and with or
        without brackets: ps.help("bars"), ps.help("chart.bars") and ps.help("bars()") all
        work. A word that charts and tables both have, such as save, explains both;
        ps.help("table.save") explains only the table one.
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


def help(word=None) -> None:  # noqa: A001 - meant to be ps.help
    """Explain one word: what it does, its options and an example you can run. ps.help("bars")

    ps.help() on its own explains how to get help. Table words work too: ps.help("keep_rows").
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
            return builtins.help(word)  # not a plotsimple word: Python's own help
        word = name
    if not keys and _old_names(word):
        old = _old_names(word)
        for name, new, key in old:
            where = f" in {_call_form(key)}" if name.endswith("=") else ""
            print(f'"{name}" is now called "{new}"{where}. The old name still works for now, but will '
                  "stop working in a later version.")
        print()
        keys = list(dict.fromkeys(key for _, _, key in old))
    if not keys:
        known = sorted({_normalise(k) for k in _all_keys()} | {a for v in _aliases().values() for a in v})
        lines = [f'plotsimple has no word called "{word}".']
        guesses = suggest(_normalise(word), known)
        if guesses:
            lines.append("Did you mean " + " or ".join(f'ps.help("{g}")' for g in guesses) + "?")
        lines.append("Type ps.words() to see every word.")
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
        forms = " or ".join(f'ps.help("{_call_form(k)[:-2]}")' for k in keys)
        print()
        print(_wrap(f'"{_normalise(word)}" can mean more than one word, so each one is explained '
                    f"above. For only one, type {forms}."))
    return None


# ---------------------------------------------------------------- ps.lesson()
# Each lesson: title, what you will learn, one or more parts, and a task.
# Each part: a sentence to start, then (line of code, what it does) pairs.
# The code of each part runs on its own (the tests run every part).
LESSONS = [
    dict(
        title="Your first chart",
        learn=[
            "how to get a table to practise with",
            "how to draw bars",
            "how to save your chart as a picture",
        ],
        parts=[
            ("A chart is made in small steps, one step on each line.", [
                ("import plotsimple as ps", 'Load plotsimple. From now on we call it by its short name, "ps".'),
                ('marks = ps.example("marks")', "Get a made-up table of test marks and call it marks."),
                ("print(marks)", "Look at the table. Each row is one child in one test. "
                                 "The names at the top (name, class, subject ...) are the columns."),
                ("chart = ps.chart(marks)", "Start a new, empty chart that uses the marks table."),
                ('chart.bars(x="name", y="marks")', "Draw bars: one bar for each name (along x, the "
                                                    "bottom), as tall as the marks (up y, the side). Each "
                                                    "child did three tests, so the bar shows the average: "
                                                    "add the three marks up and divide by 3."),
                ('chart.save("lesson1.png")', "Save the chart as a picture called lesson1.png, in the "
                                              "folder you are working in. Open it to see your chart!"),
            ]),
        ],
        notes=[
            "In Jupyter, VS Code or Spyder, write chart on the last line of a cell and the chart "
            "appears by itself. In a script, chart.show() opens it in a window.",
            'chart.save("lesson1.pdf") saves a PDF instead: the end of the name decides the type.',
            "ps.example() on its own shows every practice table and its columns.",
            "Are the names under the bars bumping into each other? Lesson 2 shows how to turn them.",
        ],
        task=[
            'Change x="name" to x="subject". What does each bar show now?',
            'Save the chart as "my_first_chart.png" and also as "my_first_chart.pdf".',
            'Make a chart of the "fruit" table: bars with x="fruit" and y="count".',
        ],
        words=["ps.example", "ps.chart", "bars", "save", "show"],
    ),
    dict(
        title="Titles and labels",
        learn=[
            "how to put a title on top",
            "how to name the x-axis and the y-axis",
            "how to add a small caption and change the size of the text",
        ],
        parts=[
            ("A chart is easier to read when it says what it shows.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.bars(x="name", y="marks")', "Draw one bar for each child."),
                ('chart.title_top("Our class marks")', "Write a title on top of the chart."),
                ('chart.label_x("Student")', "Write a name under the x-axis (the bottom)."),
                ('chart.label_y("Average mark")', "Write a name beside the y-axis (the side)."),
                ('chart.rotate_labels_x(45)', "Turn the names under the bars by 45 degrees, so "
                                              "long names do not bump into each other."),
                ('chart.caption_bottom("Made-up data from ps.example")', "Write a small line under "
                                                                         "the whole chart."),
                ("chart.font_size(14)", "Make all the text a bit bigger (normal is 12)."),
                ('chart.save("lesson2.png")', "Save it."),
            ]),
        ],
        notes=[
            "The order of the steps does not matter: you can write the title first and the bars "
            "last. plotsimple puts everything in the right place.",
            "Words that belong to one axis end in _x or _y: label_x, label_y, rotate_labels_x.",
        ],
        task=[
            'Give the chart your own title, and label the y-axis "Marks out of 100".',
            "Try chart.font_size(20). Then try 9. Which is easier to read?",
            "Take away the line chart.rotate_labels_x(45). What happens to the names?",
        ],
        words=["title_top", "label_x", "label_y", "rotate_labels_x", "caption_bottom", "font_size"],
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
            ('colour= inside a step splits the data into groups.', [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.bars(x="subject", y="marks", colour="class")', "For each subject, draw one "
                                                                       "bar per class, each class in "
                                                                       "its own colour."),
                ('chart.colours("orange", "teal", "purple")', "Choose the colours yourself, in order: "
                                                              "class A orange, B teal, C purple."),
                ("chart.legend_bottom()", "Put the legend under the chart."),
                ('chart.title_top("Marks in each subject")', "Add a title."),
                ('chart.save("lesson3_groups.png")', "Save it."),
            ]),
            ("Without groups, one colour is used for everything. You can choose it.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('weather = ps.example("weather")', "Get the weather table: three made-up cities."),
                ("chart = ps.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="rain")', "Draw a line of rain through the months. With "
                                                     "three cities in each month, the line shows "
                                                     "their average."),
                ('chart.colour("blue")', "Make the line blue."),
                ('chart.style("school")', 'Change the whole look. Try "paper", "school", '
                                          '"presentation" and "dark".'),
                ('chart.save("lesson3_look.png")', "Save it."),
            ]),
        ],
        notes=[
            'colour="class" (a column) gives each group a colour. colour="red" (a colour) makes '
            "that step red. color works too.",
            "The legend can go on any side: legend_top, legend_bottom, legend_left, legend_right. "
            "hide_legend takes it away and show_legend brings it back.",
        ],
        task=[
            'In the weather chart, add colour="city" to chart.lines(...). How many lines do you get?',
            "Move the legend to the top with chart.legend_top().",
            'Try chart.style("dark") and choose three colours that look good on it.',
        ],
        words=["colours", "colour", "legend_bottom", "legend_top", "hide_legend", "style", "lines"],
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('weather = ps.example("weather")', "Get the weather table."),
                ("chart = ps.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="temperature", colour="city", dots=True)',
                 "One line per city. dots=True also marks each month with a dot."),
                ('chart.save("lesson4_lines.png")', "Save it."),
            ]),
            ("Dots show if two things go together. This is also called a scatter chart.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks", colour="class")', "One dot for each test: how "
                                                                     "long the child studied, and "
                                                                     "the mark they got."),
                ('chart.save("lesson4_dots.png")', "Save it. Do more hours give more marks?"),
            ]),
            ("A histogram counts how many values fall in each range, like 40 to 50, 50 to 60 ...", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.histogram(x="marks", bins=8)', "Cut the marks into 8 ranges (bins) and "
                                                       "draw a bar for how many marks are in each."),
                ('chart.save("lesson4_histogram.png")', "Save it."),
            ]),
            ("Boxes show how spread out the values are. Two chart types on one chart sit on top of "
             "each other.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.boxes(x="class", y="marks")', "One box per class. The line in the box is the "
                                                      "median (the middle mark). The box holds the "
                                                      "middle half of the marks."),
                ('chart.all_dots(x="class", y="marks")', "Also show every single mark as a dot, "
                                                         "on top of the boxes."),
                ('chart.save("lesson4_boxes.png")', "Save it."),
            ]),
            ("A pie shows how a whole is shared out.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('fruit = ps.example("fruit")', "Get the fruit table: what two shops sold in a week."),
                ("chart = ps.chart(fruit)", "Start a chart."),
                ('chart.pie(labels="fruit", values="count")', "One slice per fruit, as big as the "
                                                              "number sold."),
                ('chart.save("lesson4_pie.png")', "Save it."),
            ]),
        ],
        notes=[
            "There are more chart types: area, violins, heatmap, polar, contour, surface and dots_3d. "
            "Type ps.words() to see them all and ps.help(\"violins\") to learn one.",
        ],
        task=[
            'Draw chart.violins(x="class", y="marks") instead of boxes. What is different?',
            'Draw a histogram of the weather "temperature" column with bins=6.',
            'Draw chart.area(x="year", y="population", colour="country") with ps.example("growth").',
        ],
        words=["lines", "dots", "histogram", "boxes", "all_dots", "pie", "violins", "area"],
    ),
    dict(
        title="Statistics: averages, error bars and fit lines",
        learn=[
            "what a bar shows when a group has many values",
            "how to add error bars, and the difference between standard deviation and standard error",
            "how to draw the straight line that fits some dots best",
        ],
        parts=[
            ("An average is all the values added up, divided by how many there are. Error bars "
             "show how far the values spread around it.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.bars(x="class", y="marks", show="average", error="standard deviation")',
                 "One bar per class, as tall as the average mark. The error bars show the standard "
                 "deviation: how far marks usually are from the average. A line is printed that "
                 "says which error bars were drawn."),
                ("chart.numbers_on_bars()", "Write the average on each bar."),
                ('chart.save("lesson5_average.png")', "Save it."),
            ]),
            ("A fit line is the straight line that goes as close as possible to all the dots.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "One dot per test: hours of study and the mark."),
                ("chart.fit_line(confidence_band=True, show_equation=True)", "Add the best straight "
                 "line (also called linear regression). confidence_band=True shades where the true "
                 "line very likely is (95 % confidence). show_equation=True writes its equation and "
                 "R² (1 means the dots sit right on the line, 0 means no pattern at all)."),
                ('chart.save("lesson5_fit.png")', "Save it."),
            ]),
            ("A density curve is a smooth histogram: high where many values are.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.density(x="marks", colour="class")', "One smooth curve for each class."),
                ('chart.save("lesson5_density.png")', "Save it."),
            ]),
        ],
        notes=[
            'show= can be "average", "median" (the middle value), "total" (all added up), '
            '"count" (how many rows), "smallest", "biggest" or "standard deviation": the same '
            "names as the table words average_of, median_of and so on.",
            'error= can be "standard deviation" (how spread out the values are), "standard error" '
            '(how sure we are of the average: it gets smaller with more values) or "confidence '
            'interval" (the 95 % range the true average is very likely in). The short forms "sd", '
            '"se" and "ci" work too. Always say in your caption which one you drew.',
            "fit_curve(degree=2) fits a curve with one bend; smooth_line() follows the dots freely.",
        ],
        task=[
            'Change show="average" to show="median". Do the bars change much?',
            'Change error="standard deviation" to error="standard error". Are the error bars '
            "bigger or smaller? Why?",
            'Draw weather dots with x="month_number", y="temperature" and add chart.fit_curve(degree=2).',
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('growth = ps.example("growth")', "Get the growth table: people in four made-up countries."),
                ("chart = ps.chart(growth)", "Start a chart."),
                ('chart.lines(x="year", y="population", colour="country")', "One line per country."),
                ("chart.log_scale_y()", "Use a log scale on the y-axis. A country that grows by the "
                                        "same share each year now gets a straight line."),
                ('chart.line_at_x(2010, label="2010")', "Draw an up-and-down line at the year 2010."),
                ('chart.label_y("People (millions)")', "Name the y-axis."),
                ('chart.save("lesson6_log.png")', "Save it."),
            ]),
        ],
        notes=[
            "Every axis word ends in _x or _y: range_x, ticks_x, log_scale_x, line_at_x, "
            "shade_between_x, flip_x ...",
            "range_y(0, None) fixes only the bottom and lets the top choose itself.",
        ],
        task=[
            "Move the pass mark to 60 and shade between 0 and 40 instead.",
            "Take away chart.log_scale_y(). Which countries look different?",
            'Try chart.flip_y() on any chart. What happens?',
        ],
        words=["range_y", "ticks_y", "line_at_y", "shade_between_y", "log_scale_y", "line_at_x", "flip_y"],
    ),
    dict(
        title="Several charts together",
        learn=[
            "how to make one small chart for each group",
            "how to put different charts side by side or in a grid",
            "how to compare every number column with every other",
        ],
        parts=[
            ("one_chart_per draws the same chart once for each group.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.fit_line()", "Add a fit line."),
                ('chart.one_chart_per("class")', "Make one small chart for each class, with the same "
                                                 "axes so they are easy to compare."),
                ('chart.save("lesson7_one_chart_per.png")', "Save it."),
            ]),
            ("Different charts can share one page.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('weather = ps.example("weather")', "Get the weather table."),
                ('a = ps.chart(weather).lines(x="month", y="temperature", colour="city")',
                 "Make a first chart, called a. Steps can also be joined with dots on one line."),
                ('b = ps.chart(weather).bars(x="month", y="rain", colour="city")',
                 "Make a second chart, called b."),
                ('page = ps.one_above_other(a, b)', "Put a above b on one page."),
                ('page.title_top("A year of weather")', "Give the whole page a title."),
                ('page.save("lesson7_page.png")', "Save the page."),
            ]),
            ("A grid holds many charts in rows and columns.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ('a = ps.chart(marks).bars(x="class", y="marks").title_top("Average")', "Chart a."),
                ('b = ps.chart(marks).histogram(x="marks").title_top("All marks")', "Chart b."),
                ('c = ps.chart(marks).dots(x="height", y="marks").title_top("Height")', "Chart c."),
                ('d = ps.chart(marks).boxes(x="subject", y="marks").title_top("Subjects")', "Chart d."),
                ("page = ps.grid(a, b, c, d, per_row=2)", "Put all four in a grid, 2 charts per row."),
                ('page.save("lesson7_grid.png")', "Save it."),
            ]),
            ("ps.pairs draws every number column against every other, in one go.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ('chart = ps.pairs(marks, colour="class")', "Hours, marks and height, each against each "
                                                            "other. The middle line shows how each "
                                                            "column is spread out."),
                ('chart.save("lesson7_pairs.png")', "Save it. Which two columns go together?"),
            ]),
        ],
        notes=[
            "ps.side_by_side(a, b) puts charts next to each other. same_y=True gives them the same "
            "y-axis, so heights can be compared fairly.",
            'chart.one_chart_per(down="class", across="subject") makes a table of small charts: '
            "the classes go down, the subjects go across. per_row=3 puts 3 small charts in a row.",
        ],
        task=[
            'Change chart.one_chart_per("class") to chart.one_chart_per("subject").',
            "Use ps.side_by_side(a, b) instead of ps.one_above_other(a, b).",
            'Make ps.pairs of the weather table with colour="city".',
        ],
        words=["one_chart_per", "ps.one_above_other", "ps.grid", "ps.pairs", "ps.side_by_side", "fit_line"],
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('weather = ps.example("weather")', "Get the weather table."),
                ("chart = ps.chart(weather)", "Start a chart."),
                ('chart.lines(x="month", y="temperature")', "A line of temperature (average of the "
                                                            "three cities)."),
                ('chart.second_y("rain", drawn_as="bars", label="Rain (mm)")', "Add rain, drawn as "
                                                                               "bars, with its own "
                                                                               "y-axis on the right."),
                ('chart.label_y("Temperature (°C)")', "Name the left y-axis."),
                ('chart.save("lesson8_second_y.png")', "Save it."),
            ]),
            ("A zoom box shows a small part of the chart close up.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.zoom_box(x_between=[2, 4], y_between=[40, 60], where=\"top left\")",
                 "Show the part between 2 and 4 hours and between 40 and 60 marks, bigger, in a "
                 "box at the top left."),
                ('chart.save("lesson8_zoom.png")', "Save it."),
            ]),
            ("A heatmap colours a grid of numbers. cluster=True puts rows that are alike next to "
             "each other.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('weather = ps.example("weather")', "Get the weather table."),
                ("chart = ps.chart(weather)", "Start a chart."),
                ('chart.heatmap(x="month", y="city", value="rain", numbers=True, cluster=True)',
                 "One square per city and month, coloured by rain, with the number written in. "
                 "cluster=True sorts the rows and columns and draws a tree of how alike they are."),
                ('chart.label_colour_bar("Rain (mm)")', "Name the colour bar."),
                ("chart.size(width=12, height=4)", "Make the chart wide and low."),
                ('chart.save("lesson8_heatmap.png")', "Save it."),
            ]),
            ("An animation is a chart that moves: one picture for each value of a column.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('growth = ps.example("growth")', "Get the growth table."),
                ("chart = ps.chart(growth)", "Start a chart."),
                ('chart.bars(x="country", y="population")', "One bar per country."),
                ("chart.range_y(0, 50)", "Keep the y-axis still, from 0 to 50."),
                ('chart.animate(over="year")', "Make one picture for each year."),
                ('chart.save("lesson8_growth.gif")', "Save it as a .gif, a moving picture. Open it "
                                                     "in a web browser to watch."),
            ]),
            ("Your own program. Every chart follows the same plan: get data, start, draw, words, "
             "look, save.", [
                ("import plotsimple as ps", "1. Load plotsimple."),
                ('island = ps.example("island")', '2. Get data. For your own file write '
                                                  'ps.chart("my_file.xlsx") instead.'),
                ("chart = ps.chart(island)", "3. Start a chart."),
                ('chart.contour(x="east", y="north", z="height")', "4. Draw: a map where each colour "
                                                                   "band is one height."),
                ('chart.title_top("Treasure Island")', "5. Words: a title ..."),
                ('chart.label_x("East (km)")', "... and the axis names ..."),
                ('chart.label_y("North (km)")', ""),
                ('chart.label_colour_bar("Height (m)")', "... and the name of the colour bar."),
                ('chart.note("X marks the spot", x=10, y=4, arrow=True)', "A note with an arrow."),
                ('chart.style("presentation")', "6. Look: big and clear for showing to others."),
                ('chart.save("lesson8_my_program.png")', "7. Save."),
            ]),
        ],
        notes=[
            "Those are all the charts! ps.words() lists every word and ps.help(\"word\") "
            "explains any of them. Lessons 9 to 12 teach tables: how to look at your data, "
            "clean it and sum it up before you draw it. Lesson 13 teaches numbers and lesson 14 "
            "statistical tests.",
            'Your own data: save it from Excel as .xlsx or .csv, then chart = ps.chart("my_file.xlsx"). '
            "If a column name is spelt wrong, plotsimple tells you the right one.",
        ],
        task=[
            "Write your own program, from an empty file, for a question you care about.",
            "Use any example table, or your own. Give it a title, axis names and a note.",
            "Put two of your charts side by side and save the page as a PDF.",
        ],
        words=["second_y", "zoom_box", "heatmap", "label_colour_bar", "size", "animate",
               "contour", "note", "style", "ps.words", "ps.help"],
    ),
    dict(
        title="Tables: open one and look at it",
        learn=[
            "how to turn data into a plotsimple table",
            "how to look at the first and last rows, and the names of the columns",
            "how to describe every column and count what is in it",
        ],
        parts=[
            ("Before you draw, look at your data. ps.table gives you plain words for that.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table from the practice marks. For "
                                                          'your own file, write ps.table("my_file.xlsx").'),
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("fruit"))', "Make a table of fruit sold by two shops."),
                ("table.show_first_rows()", "Show the first rows. With nothing in the brackets you "
                                            "get 5."),
                ('table.count_each("fruit")', "For each fruit: in how many rows it comes up. It says the "
                                              "answer in a sentence, and gives back a small table that "
                                              "also has the percent of all the rows."),
                ('table.count_each("shop")', "The same for each shop."),
            ]),
        ],
        notes=[
            "In Jupyter, VS Code or Spyder, write table on the last line of a cell to see it as a "
            "nice table.",
            "ps.table reads the same files as ps.chart: .xlsx, .csv, .tsv, .parquet and .json.",
            "The row numbers on the left start at 0, as everywhere in Python.",
        ],
        task=[
            'Make a table of ps.example("weather"). How many rows does it have?',
            "Describe its columns. Which column has the biggest numbers?",
            'Count each "city". Does every city have the same number of rows?',
        ],
        words=["ps.table", "table.show_first_rows", "table.show_last_rows", "table.column_names",
               "table.describe_columns", "table.count_rows", "table.count_each"],
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.keep_columns("name", "subject", "marks")', "Keep only these three columns, "
                                                                   "in this order."),
                ('table.keep_rows("subject", equal_to="Maths")', "Keep only the rows where the subject "
                                                                 "is Maths. plotsimple says how many "
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("weather"))', "Make a table of the weather."),
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
        ],
        notes=[
            "Conditions for keep_rows and remove_rows: above, below, at_least, at_most, equal_to, "
            "not_equal_to, between, one_of, contains, starts_with, ends_with and is_empty.",
            'one_of= takes a list: table.keep_rows("subject", one_of=["Maths", "Science"]).',
            "undo() can take back many steps, one at a time. Your file is never changed: only "
            "table.save(...) writes a file.",
        ],
        task=[
            'In the marks table, keep only class "B" and sort by "hours". Who studied the longest?',
            "Undo the sorting, then sort by name instead.",
            'Keep 5 random rows with table.keep_random_rows(5, seed=1) and save them as "five.csv".',
        ],
        words=["table.keep_columns", "table.keep_rows", "table.sort_by", "table.steps", "table.undo",
               "table.start_again", "table.remove_rows", "table.remove_columns", "table.rename_column",
               "table.save"],
    ),
    dict(
        title="Tables: clean the data and add new columns",
        learn=[
            "how to find and fill empty cells",
            "how to fix columns: numbers written as text, extra spaces, wrong values",
            "how to add new columns, in plain words or like a sum in maths",
        ],
        parts=[
            ("Real data often has empty cells. The practice tables have none, so we make some first.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table({"name": ["  Asha", "ben ", "Chen"], "height": ["144", "152", "unknown"]})',
                 "Make a small table by hand: each column name, then its values in a list."),
                ('table.remove_extra_spaces("name")', "Take away the spaces before and after the names."),
                ('table.replace_values("name", "ben", with_value="Ben")', 'Write "ben" with a capital '
                                                                          "letter."),
                ('table.make_numbers("height")', "Turn the heights from text into numbers. "
                                                 '"unknown" is not a number, so that cell becomes '
                                                 "empty, and plotsimple tells you."),
                ("table.describe_columns()", "Now height is a column of numbers."),
            ]),
            ("A new column can be worked out from other columns, in plain words.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.add_column("marks_per_hour", by_dividing=["marks", "hours"])',
                 'A new column called "marks_per_hour": the marks divided by the hours.'),
                ('table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], '
                 'names=["D", "C", "B", "A"])',
                 "A grade for each mark: below 40 is D, 40 to 59 is C, 60 to 79 is B, 80 and more is A."),
                ("table.show_first_rows()", "Look at the two new columns on the right."),
                ('table.count_each("grade")', "How many of each grade?"),
            ]),
            ("Or write the new column like a sum in maths. Here each subject first gets its own "
             "column.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
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
        ],
        notes=[
            'fill_empty_cells can also use with_the="median", "value above" or "value below", or '
            "with_value=0 for a number you choose. Filling with the average makes the values look "
            "less spread out than they really are, so say in your report how you filled them.",
            "remove_rows_with_empty_cells() with nothing in the brackets removes every row that has "
            "an empty cell in any column.",
            "More fixing words: make_text, make_dates, make_lowercase and make_uppercase.",
            "Plain words for new columns: by_adding=, by_subtracting=, by_multiplying= and "
            "by_dividing=, or from_column= with plus=, minus=, times= or divide_by=.",
            'In a maths-style column, a column name with spaces goes in backticks: "`max temp` - 2".',
        ],
        task=[
            "In the first example, fill the empty marks with the median instead of the average.",
            'Add a column "height_m" with the height in metres: from_column="height", divide_by=100.',
            'In the last example, add a column "best_two" with the formula "total - English".',
        ],
        words=["table.replace_values", "table.count_empty_cells", "table.fill_empty_cells",
               "table.remove_rows_with_empty_cells",
               "table.remove_extra_spaces", "table.make_numbers", "table.add_column",
               "table.add_group_column", "table.add_rank_column", "table.turn_rows_into_columns"],
    ),
    dict(
        title="Tables: summaries, and from a table to a chart",
        learn=[
            "how to work out averages, totals and counts, for each group",
            "how to tell if two columns go together",
            "how to draw a chart of a summary",
            "how to join two tables and turn columns into rows",
        ],
        parts=[
            ("A summary turns many rows into a few numbers. It never changes your table, and it "
             "says its answer in a sentence.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.average_of("marks")', "The average of all the marks: just one number."),
                ('table.average_of("marks", for_each="class")', "One average for each class. for_each= "
                                                                "also gives back a new, small table."),
                ('table.biggest_of("marks", for_each="subject")', "The best mark in each subject."),
                ('table.correlation_between("hours", "marks")', "Do more hours go with more marks? "
                                                                "The answer is a number r from -1 "
                                                                "to 1, explained in words."),
            ]),
            ("A summary is a table too, so you can sort it and draw it.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("fruit"))', "Make a table of fruit sales."),
                ('totals = table.total_of("count", for_each="fruit")', "A new table: one row per fruit, "
                                                                      'with a column "total of count".'),
                ('totals.sort_by("total of count", biggest_first=True)', "Best seller first."),
                ("chart = totals.chart()", "Start a chart from the summary table."),
                ('chart.bars(x="fruit", y="total of count")', "One bar per fruit."),
                ("chart.numbers_on_bars()", "Write the total on each bar."),
                ('chart.save("lesson12_fruit.png")', "Save the chart."),
            ]),
            ("Two tables can be joined: their rows are matched on a column they share.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('ages = {"name": ["Asha", "Ben", "Chen"], "age": [12, 13, 12]}', "A small table of "
                                                                                  "ages, typed by hand: "
                                                                                  "each column name, "
                                                                                  "then its values."),
                ('table.join_with(ages, using="name")', "Bring in the age, matching the rows on the "
                                                        "name. Only children found in both tables stay."),
                ("table.show_first_rows()", "Look: there is a new column, age."),
                ('table.average_of("marks", for_each="age")', "The average mark at each age."),
            ]),
            ("Turning columns into rows gives the long shape that many charts like.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table({"name": ["Asha", "Ben", "Chen"], "test_1": [61, 73, 81], '
                 '"test_2": [70, 68, 90]})', "A small table: two tests, one column for each."),
                ('table.turn_columns_into_rows(["test_1", "test_2"], names_to="test", values_to="mark")',
                 'Put the two tests under each other. The new column "test" says which test, and '
                 '"mark" holds the mark. Each child now has two rows.'),
                ("table.show_first_rows(6)", "Look at the new shape."),
                ("chart = table.chart()", "Start a chart from it."),
                ('chart.bars(x="name", y="mark", colour="test")', "Both tests side by side for each "
                                                                  "child."),
                ('chart.save("lesson12_tests.png")', "Save it."),
            ]),
        ],
        notes=[
            "More summaries: median_of, total_of, smallest_of, biggest_of, standard_deviation_of "
            "and count_rows. Each one can take for_each=.",
            "A correlation does not show that one thing causes the other.",
            "table.add_rows_from(other) puts the rows of another table under this one. "
            "turn_rows_into_columns does the opposite of turn_columns_into_rows.",
            "table.as_pandas() (or table.as_polars()) gives back a plain pandas (or Polars) "
            "table, if another Python tool needs one. plotsimple words never need it.",
            "Lesson 13 teaches numbers, and lesson 14 the statistical tests: is a difference "
            "real, or could it be chance?",
        ],
        task=[
            'Make the fruit totals for_each="shop" instead. Which shop sold more?',
            'Work out the average temperature for_each="city" in the weather table and draw it as bars.',
            "Is there a link between height and marks? Use correlation_between to find out.",
        ],
        words=["table.average_of", "table.biggest_of", "table.correlation_between", "table.total_of",
               "table.chart", "table.join_with", "table.turn_columns_into_rows", "table.add_rows_from",
               "table.as_pandas"],
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
                ("import plotsimple as ps", "Load plotsimple."),
                ("nums = ps.numbers([12, 15, 9, 20, 14])", "Make a list of five numbers and call it nums."),
                ("print(nums)", "Look at them. The first line says how many there are."),
                ("bigger = nums * 2", "Maths with symbols makes NEW numbers: every number times 2. "
                                      "nums itself does not change."),
                ("print(bigger)", "24, 30, 18, 40 and 28."),
                ("nums.add(1)", "Maths with words CHANGES the numbers you have: add 1 to each one."),
                ("nums.undo()", "Like tables, every word can be taken back."),
                ("nums.keep_numbers(above=10)", "Keep only the numbers above 10. The same conditions "
                                                "as table.keep_rows: above, below, between ..."),
                ("nums.average()", "The average of what is left, said in a sentence."),
                ("nums.standard_deviation()", "How spread out they are (divide by n - 1, as in pandas "
                                              "and R)."),
                ("nums.steps()", "List the steps you did."),
            ]),
            ("Counting, random numbers and a chart.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ("months = ps.numbers(1, to=12)", "Count from 1 to 12. The end is included."),
                ("temps = ps.numbers([4, 6, 9, 13, 17, 20, 22, 21, 17, 12, 8, 5])", "A temperature for "
                                                                                    "each month."),
                ("chart = ps.chart()", "Start an empty chart."),
                ("chart.lines(x=months, y=temps, dots=True)", "Draw the temperatures as a line, month by "
                                                              "month."),
                ('chart.save("lesson13_line.png")', "Save it."),
                ("heights = ps.random_numbers(how_many=200, average=150, standard_deviation=8, seed=1)",
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
                ("import plotsimple as ps", "Load plotsimple."),
                ("grid = ps.number_grid([[3, 5, 2], [8, 6, 4]])", "A grid with 2 rows and 3 columns: one "
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
            "Make 6 dice throws with ps.random_numbers(how_many=6, between=[1, 6], whole_numbers=True, "
            "seed=1). What is their total?",
            "Make a grid of 3 rows and 4 columns filled with 0: ps.number_grid(rows=3, columns=4, "
            "filled_with=0).",
        ],
        words=["ps.numbers", "ps.random_numbers", "ps.number_grid", "nums.add", "nums.undo",
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
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.compare_averages("marks", between="class")',
                 "Do the three classes really differ in their average mark? The answer comes in "
                 "words. The p-value is the chance of a difference this big if the classes were "
                 "really the same: below 0.05 we call it unlikely to be chance. With three or more "
                 "groups this test is also called Welch's ANOVA; with two groups, a t-test."),
                ('table.compare_each_pair("marks", between="class")',
                 "Which classes differ? Each pair gets its own t-test (Welch's). Testing many pairs makes a "
                 "false alarm more likely, so the p-values are adjusted (Holm's method)."),
                ('table.compare_averages("marks", between="subject", based_on="ranks")',
                 "based_on=\"ranks\" puts the marks in order and compares the places instead "
                 "(a Kruskal-Wallis test, or Mann-Whitney for two groups). It is safer when groups "
                 "are small and the values are not bell-shaped."),
            ]),
            ("Counts can be compared too: does how often something happens differ between groups?", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.add_group_column("result", from_column="marks", cut_at=[60], '
                 'names=["below 60", "60 or more"])', "A new column that says if each mark is below "
                                                      "60 or not."),
                ('table.compare_counts("result", between="class")',
                 "Do the classes differ in how many marks are 60 or more? This counts the rows for "
                 "each class and result and uses the chi-square test. Cramer's V says how strong "
                 "the link is, from 0 (none) to 1."),
            ]),
            ("A fit finds the line or formula that follows the data best, and says how sure it is.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('table = ps.table(ps.example("marks"))', "Make a table of the marks."),
                ('table.fit_straight_line(x="hours", y="marks")',
                 "How much do the marks go up for each extra hour of study? This is linear regression "
                 "by least squares. R squared says how much of the differences in marks the line "
                 "explains."),
                ('growth = ps.table(ps.example("growth"))', "Make a table of the growth data."),
                ('growth.keep_rows("country", equal_to="Applia")', "Keep one country."),
                ('growth.fit_formula("population = a * exp(b * (year - 2000))")',
                 "Fit a formula you write yourself. Names that are columns (population, year) are "
                 "data; the others (a and b) are found by plotsimple. Here b is the growth rate "
                 "each year. This is nonlinear least squares."),
            ]),
        ],
        notes=[
            "A p-value says how surprising the data would be if there were no real difference. "
            "It does not say how big or important the difference is: the effect size does that.",
            "A large p-value does not show that there is no difference. It only means these data "
            "cannot tell.",
            "Each test gives back a one-row table with all its numbers, so you can save it: "
            'result = table.compare_averages(...), then result.save("test.csv").',
            "Also: compare_before_after for two measurements of the same thing (a paired t-test), "
            "and compare_average_to for one average against a number you choose.",
        ],
        task=[
            'Compare the average marks between="subject". Could the differences be chance?',
            'Is the average temperature different between the cities of ps.example("weather")?',
            'Fit a straight line of "forest" against "year" in ps.example("growth") for one country.',
        ],
        words=["table.compare_averages", "table.compare_each_pair", "table.compare_counts",
               "table.fit_straight_line", "table.fit_formula", "table.add_group_column"],
    ),
    dict(
        title="Tell a whole story",
        learn=[
            "how to do a whole analysis with one object, a story: open, clean, sum up, draw and save",
            "why chart types start with draw_ in a story, and which chart the other chart words change",
            "how undo() and steps() cover table steps and chart steps together",
            "how to put several charts on one page with letters, for a paper",
        ],
        parts=[
            ("A story is one object for the whole analysis. Every table word and every chart word "
             "works on it, so the program reads from top to bottom like a short story.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('story = ps.start(example="marks")', "Open a table and start a story with it. For your own "
                                                      'file, write ps.start(file="results.xlsx").'),
                ("story.show_first_rows(5)", "Look at the first 5 rows before you change anything."),
                ("story.remove_rows_with_empty_cells()", "Clean: remove the rows with an empty cell. The "
                                                         "practice table has none, so nothing is removed."),
                ('story.keep_rows("subject", equal_to="Maths")', "Keep only the Maths marks."),
                ('story.average_of("marks", for_each="class")', "Sum up: the average Maths mark of each "
                                                                 "class, said in a sentence."),
                ('story.draw_boxes(x="class", y="marks")', "Now the chart part of the story begins. Chart "
                                                           "types start with draw_, so every line starts with "
                                                           "a verb. This is chart 1."),
                ('story.draw_all_dots(x="class", y="marks")', "Draw every mark as a dot on top of the boxes: "
                                                              "a second layer of the same chart."),
                ('story.title_top("Maths marks in each class")', "Words such as title_top change the chart you "
                                                                 "drew last."),
                ('story.line_at_y(50, label="pass mark")', "A line at the pass mark."),
                ("story.new_chart()", "Start another chart. The next draw_ step draws it."),
                ('story.draw_dots(x="hours", y="marks", colour="class")', "Chart 2: hours of study against "
                                                                          "marks, one colour per class."),
                ("story.fit_line(confidence_band=True)", "A straight line for each class (linear regression), "
                                                         "with a band for how sure the line is."),
                ('story.label_x("Hours of study")', "The name under the x-axis of chart 2."),
                ("story.put_charts_together(per_row=2, letters=True)", "Both charts on one page, 2 in a row, "
                                                                       "with A and B at the top left, as "
                                                                       "figures in papers have."),
                ('story.save_chart("lesson15_figure.png")', "Save the page. save_chart saves the chart, or the "
                                                            "page once the charts are put together."),
                ('story.save_table("lesson15_maths.xlsx")', "Save the cleaned table too: only the Maths rows."),
                ("story.steps()", "The whole story, table steps and chart steps in one list."),
            ]),
            ("Summaries give back a new story, so you can draw them. undo() takes back any step.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('story = ps.start(example="fruit")', "A story about fruit sold in two shops."),
                ('totals = story.total_of("count", for_each="fruit")', "The total count of each fruit. It "
                                                                       "says the answer, and gives back a new "
                                                                       "story called totals."),
                ('totals.sort_by("total of count", biggest_first=True)', "Sort the totals, biggest first."),
                ('totals.draw_bars(x="fruit", y="total of count")', "Draw the totals as bars."),
                ("totals.numbers_on_bars()", "Write the number on each bar."),
                ('totals.save_chart("lesson15_totals.png")', "Save the chart."),
                ('story.keep_rows("shop", equal_to="Corner shop")', "Back to the first story: keep one shop. "
                                                                    "totals does not change."),
                ('story.draw_dots(x="price", y="count")', "Draw price against how many were sold."),
                ("story.undo()", "Undo takes back the last step, whether it changed the table or the chart: "
                                 "here the dots."),
                ("story.undo()", "And again: the shop rows come back."),
                ("story.steps()", "Nothing is left in the list: both steps were taken back."),
            ]),
        ],
        notes=[
            "A chart uses the table as it was when its first draw_ step was written. Table steps "
            "written after it do not change it, so the story reads from top to bottom. For the "
            "table as it is now, start a new chart with story.new_chart().",
            "save_table saves the table and save_chart the chart. Plain save works too, because the "
            "file type says which: .xlsx or .csv is a table, .png or .pdf is a picture.",
            "Charts from different tables go on one page with ps.grid(story_a, story_b, letters=True): "
            "each story gives the chart it drew last.",
            "ps.table(...) and ps.chart(...) are still there underneath, for programmers who want a "
            "table and a chart as two separate objects.",
        ],
        task=[
            'Tell a story about ps.start(example="weather"): keep one city, draw the temperature as '
            "lines, give it a title and save the chart and the table.",
            "Add a second chart of the rain with new_chart(), then put the two charts together with "
            "letters=True.",
            "Write a table step after your first chart, then undo() twice. Look at steps() after "
            "each undo.",
        ],
        words=["ps.start", "story.draw_boxes", "story.draw_all_dots", "story.draw_dots", "story.draw_bars",
               "story.new_chart", "story.put_charts_together", "story.save_chart", "story.save_table",
               "story.steps", "story.undo"],
    ),
]


def _wrap(text: str, indent: str = "", width: int = WIDTH) -> str:
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def lesson_code(number: int) -> list[str]:
    """The code of each part of a lesson, ready to run (used by the tests)."""
    return ["\n".join(code for code, _ in steps) for _, steps in LESSONS[number - 1]["parts"]]


def _list_lessons() -> None:
    print(f"Learn plotsimple in one day: {len(LESSONS)} lessons of about 30 minutes.")
    print("Lessons 1 to 8 teach charts, lessons 9 to 12 teach tables, lesson 13 teaches numbers,")
    print("lesson 14 teaches statistical tests and lesson 15 tells a whole story with all of them.\n")
    for i, les in enumerate(LESSONS, start=1):
        print(f"  ps.lesson({i}){' ' * (5 - len(str(i)))}{les['title']}")
    print("\nStart with ps.lesson(1). Each lesson ends with a small task to try yourself.")


def lesson(number=None) -> None:
    """A lesson: an example to run, what each line does, and a task. ps.lesson(1)

    ps.lesson() on its own lists all the lessons.
    """
    if number is None:
        _list_lessons()
        return None
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= len(LESSONS):
        raise PlotsimpleError(
            f"There are lessons 1 to {len(LESSONS)}, so ps.lesson({number!r}) does not exist.\n"
            "Start with ps.lesson(1), or type ps.lesson() to see them all."
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
    print()
    if number < len(LESSONS):
        print(f"Next: ps.lesson({number + 1})  {LESSONS[number]['title']}")
    else:
        print("You have finished all the lessons. Well done!")
    return None
