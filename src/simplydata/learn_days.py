"""30 days of simplydata: the days, their warm-up questions, tasks and answers.

This file is the one source for both sdata.today() (and its family: start_30_days, day, answers,
next_day) and the printable guide docs/manual/simplydata_30_days.pdf (made by
docs/make_30_days.py). The tests run every piece of code in it and check that the guide matches.

There are two tracks of 30 days, about an hour a day:
  "python"    for people who already write some Python: about two lessons a day, then an
              eight-day project of their own;
  "beginner"  for people who have never programmed: three days of just enough Python first, then
              about one lesson a day and a small project.

A day is made of units (UNITS below). A unit belongs to a lesson (sdata.lesson(n)) or is a day of
its own (Python first steps, review, project, final test). Each unit has recall questions about its
own words. The warm-up of a day is not written by hand: it asks one question from yesterday, one
from about three days ago, one from about a week ago (and two weeks ago, later on). Taking a short
test is a better way to remember than reading again (retrieval practice, Roediger & Karpicke 2006),
and spreading it over days works better than all at once (spacing, Cepeda et al. 2006).
"""

from __future__ import annotations

import datetime
import json
import os
import textwrap
from pathlib import Path

from ._errors import SimplydataError, suggest

TRACK_NAMES = ("python", "beginner")
DAYS_IN_A_TRACK = 30

TRACKS = {
    "python": dict(
        title="Python track",
        who="for people who already write some Python",
        about="About two lessons a day for 20 days, with review days on days 7, 14, 21 and 28, then an "
              "eight-day project of your own (days 21 to 28), a final test from memory and a look at the "
              "same work in pandas and matplotlib.",
    ),
    "beginner": dict(
        title="Beginner track",
        who="for people who have never programmed",
        about="Three days of just enough Python first, then about one lesson a day, with review days on "
              "days 7, 14 and 21, a small project of your own (days 27 and 28), a final test from memory "
              "and a look back.",
    ),
}


def _code(text: str) -> str:
    return textwrap.dedent(text).strip()


def _lesson_do(number: int, which: str = "") -> str:
    part = f", {which}" if which else ""
    return (f"Work through sdata.lesson({number}){part}: type each example yourself (do not copy and paste), "
            f"run it and read what each line does. Then do the lesson's tasks and check them with "
            f"sdata.lesson({number}, solutions=True).")


# ---------------------------------------------------------------- the units
# Each unit: lessons (sdata.lesson numbers), manual (extra sections: (label in the manual, title)),
# words (keys of learn.GROUPS to look up with sdata.help), do (what to do), examples (blocks of
# (code, what it does) that run on their own), errors ((code that fails on purpose, what the error
# says, what it means)), recall ((question, answer)), task and solutions (one (code, explanation)
# per task; the code runs on its own, or is "" for a task that is written, not run).
UNITS = {
    # ------------------------------------------------------------ Python first steps (beginner)
    "py_values": dict(
        lessons=[],
        manual=[("start:install", "Install and first steps")],
        words=[],
        do=["Open a place to type Python: a Jupyter notebook, VS Code, Spyder, or IDLE (it comes with "
            "Python). The manual's \"Install and first steps\" says how.",
            "Type each line of the example below, run it, and read what it prints.",
            "Then save the same lines in a file called day1.py and run the whole file: in a terminal, "
            "type python day1.py. A file of lines like this is called a script."],
        examples=[[
            ('name = "Asha"', 'Give the text "Asha" the name name. Text always goes in quotes.'),
            ("age = 12", "Give the number 12 the name age. Numbers have no quotes."),
            ("print(name)", "print shows a value on the screen: Asha."),
            ("print(age + 1)", "Maths works as on a calculator: + - * / and ** for a power. This prints 13."),
            ('print("Hello", name)', "print can show several values, separated by commas."),
            ("height_m = 1.44", "A name has no spaces (use _ instead), and a decimal number has a point."),
            ("print(height_m * 100)", "Prints 144.0: the height in centimetres."),
        ]],
        recall=[
            ("How do you write the text Asha in Python?",
             '"Asha", in quotes. Without quotes, Python thinks Asha is a name.'),
            ("How do you give the number 12 the name age?",
             "age = 12. The name goes on the left of =, the value on the right."),
            ("What does print(age) do?", "It shows the value of age on the screen."),
        ],
        task=['Give the text "Bergen" the name city and the number 2250 the name rain (the rain in a year, '
              'in mm). Print the words "Rain in", the city, and the rain of an average month (rain / 12).'],
        solutions=[(_code('''
            city = "Bergen"
            rain = 2250
            print("Rain in", city, rain / 12)'''),
            "It prints: Rain in Bergen 187.5. Text in quotes, numbers without, and print with commas "
            "between the values.")],
    ),
    "py_lists": dict(
        lessons=[],
        manual=[("start:install", "Install and first steps")],
        words=["sdata.numbers", "nums.average", "nums.keep_numbers"],
        do=["Type and run the example below, one line at a time.",
            "Look closely at the brackets: square brackets [ ] make a list, round brackets ( ) call a "
            "word, and curly brackets { } make a table by hand.",
            'Then type sdata.help("average") and read what it says.'],
        examples=[[
            ("marks = [48, 71, 55, 88, 62]", "A list: values in square brackets, with commas between."),
            ("print(len(marks))", "len is a word of Python that counts: 5. A word is called with round "
                                  "brackets around what it works on."),
            ("print(round(3.14159, 2))", "Two things in the brackets, separated by a comma: round 3.14159 "
                                         "to 2 decimals."),
            ("import simplydata as sdata", 'Load simplydata. From now on we call it by its short name, sdata.'),
            ("nums = sdata.numbers(marks)", "numbers is a word of sdata: the point joins a thing and its "
                                            "word. We call the result nums."),
            ("nums.average()", "A step on the numbers: it prints the answer in a sentence. Empty brackets "
                               "still have to be there."),
            ("nums.keep_numbers(above=50)", "above=50 is an option with a name: the name, =, the value."),
            ("nums.show()", "Show the numbers that are left."),
            ('kids = {"name": ["Asha", "Ben"], "height": [144, 152]}', "A table typed by hand: each column "
                                                                      "name with its list of values."),
            ("table = sdata.table(kids)", "Turn it into a simplydata table."),
            ("table.show()", "Show the table."),
        ]],
        recall=[
            ("How do you write a list of the numbers 3, 5 and 2?", "[3, 5, 2]: square brackets, commas between."),
            ("In nums.keep_numbers(above=50), what is above=50?",
             "An option with a name: the name, =, and the value, inside the round brackets."),
            ("What does every simplydata program start with?", "import simplydata as sdata"),
            ("Which word gives the average of numbers called nums?", "nums.average()"),
        ],
        task=["Make a list of five temperatures, turn it into simplydata numbers, and find the biggest one "
              "and the median."],
        solutions=[(_code('''
            import simplydata as sdata
            temperatures = [4, 9, 13, 17, 20]
            nums = sdata.numbers(temperatures)
            nums.biggest()
            nums.median()'''),
            "The biggest is 20 and the median (the middle value when they are in order) is 13.")],
    ),
    "py_errors": dict(
        lessons=[],
        manual=[("start:install", "Install and first steps")],
        words=["sdata.help"],
        do=["Errors are normal: everybody gets them, every day. Today you make some on purpose.",
            "Type each of the errors to make on purpose, below, and read the error from the bottom up: "
            "the last line says what went wrong.",
            "Then start lesson 1, below."],
        errors=[
            ("print(nmae)", "is not defined",
             "NameError: a name is misspelt (nmae, not name), or was never given a value with = first."),
            ('print("Asha)', "unterminated string",
             "SyntaxError: a quote mark or a bracket is not closed. Every opening quote mark needs a closing "
             "one, and so does every bracket."),
            ('sdata.example("wether")', 'Did you mean sdata.example("weather")',
             "simplydata's own errors say what went wrong and suggest the word you probably meant."),
            ("nums = sdata.numbers([3, 5, 2])\nnums.avrage()", "Did you mean .average()",
             "A misspelt step: simplydata lists the steps you can use and the nearest one."),
        ],
        recall=[
            ("Where in an error message do you look first?",
             "At the last line: it names the kind of error and says what went wrong."),
            ("What does NameError: name 'nmae' is not defined usually mean?",
             "A name is misspelt, or was never given a value (with =) before it was used."),
            ("Python says SyntaxError. What do you check first?",
             "Quote marks and brackets: every one that opens must be closed."),
        ],
        task=['Find and fix the two mistakes in these lines, then run them: '
              'nums = sdata.numbers([4, 8, 6) and then nums.averag()'],
        solutions=[(_code('''
            import simplydata as sdata
            nums = sdata.numbers([4, 8, 6])
            nums.average()'''),
            "The list needs its closing ] before the ), and the word is average. The average is 6.")],
    ),
    # ------------------------------------------------------------ charts (lessons 1 to 8)
    "lesson1": dict(
        lessons=[1],
        words=["sdata.example", "sdata.chart", "bars", "save"],
        recall=[
            ("Which word lists the practice tables?", 'sdata.example(); sdata.example("marks") gets one of them.'),
            ("How do you start a chart of a table called marks?", "chart = sdata.chart(marks)"),
            ("Which word draws one bar for each name, as tall as the marks?", 'chart.bars(x="name", y="marks")'),
            ("How do you save a chart as a PDF?",
             'chart.save("my_chart.pdf"): the end of the file name decides the type.'),
        ],
        task=['Make a bar chart of the "weather" table with x="city" and y="rain", and save it as "rain.png".'],
        solutions=[(_code('''
            import simplydata as sdata
            weather = sdata.example("weather")
            chart = sdata.chart(weather)
            chart.bars(x="city", y="rain")
            chart.save("rain.png")'''),
            "One bar for each city, as tall as its average rain over the 12 months.")],
    ),
    "lesson2": dict(
        lessons=[2],
        words=["title_top", "label_x", "label_y", "rotate_labels_x", "caption_bottom", "font_size"],
        recall=[
            ("Which word puts a title above the chart?", 'chart.title_top("My title")'),
            ("How do you name the y-axis?", 'chart.label_y("Marks out of 100")'),
            ("The names under the bars bump into each other. Which word helps?", "chart.rotate_labels_x(45)"),
            ("Which word makes all the text on a chart bigger or smaller?", "chart.font_size(14)"),
        ],
        task=['Draw dots of "hours" (x) against "marks" (y) from the marks table, with a title, a name for '
              "each axis and a caption at the bottom."],
        solutions=[(_code('''
            import simplydata as sdata
            marks = sdata.example("marks")
            chart = sdata.chart(marks)
            chart.dots(x="hours", y="marks")
            chart.title_top("More study, better marks?")
            chart.label_x("Hours of study")
            chart.label_y("Mark (out of 100)")
            chart.caption_bottom("Made-up data")
            chart.save("hours_and_marks.png")'''),
            "First draw, then add the words; the order of the word steps does not matter.")],
    ),
    "first_table": dict(
        lessons=[],
        manual=[("start:welcome", "Welcome")],
        words=["sdata.table", "table.keep_rows", "table.average_of", "table.chart"],
        do=["Run the example below: your first table. A table and a chart are separate things: one clean "
            "table can feed many charts. Tables get their own lessons on day 6."],
        examples=[[
            ("import simplydata as sdata", "Load simplydata."),
            ('table = sdata.table(sdata.example("marks"))', "Start a table from the marks practice table."),
            ('table.keep_rows("marks", above=50)', "Keep only the rows with marks above 50. It says how many "
                                                    "rows are left."),
            ('table.average_of("marks", for_each="class")', "The average mark of each class, in a sentence."),
            ("chart = table.chart()", "Start a chart from the table as it is now."),
            ('chart.bars(x="class", y="marks")', "One bar for each class."),
            ('chart.save("good_marks.png")', "Save it."),
        ]],
        recall=[
            ("How do you start a table from the marks practice table?", 'table = sdata.table(sdata.example("marks"))'),
            ("Which word keeps only the rows with marks above 50?", 'table.keep_rows("marks", above=50)'),
            ("Which word gives the average mark of each class?", 'table.average_of("marks", for_each="class")'),
        ],
        task=['Start a table of the weather, keep the rows with "temperature" above 15, and give the '
              'average "rain" for each city.'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("weather"))
            table.keep_rows("temperature", above=15)
            table.average_of("rain", for_each="city")'''),
            "keep_rows says how many warm months are left; average_of gives one average for each city.")],
    ),
    "lesson3": dict(
        lessons=[3],
        words=["colour", "colours", "legend_bottom", "order_x", "style"],
        recall=[
            ("How do you give each class its own colour in a bar chart?",
             'colour="class" inside the chart step: chart.bars(x="subject", y="marks", colour="class")'),
            ("Which word moves the legend below the chart?", "chart.legend_bottom()"),
            ("Which word lets you choose the colours of the groups?", 'chart.colours("orange", "teal", "purple")'),
            ("How do you put the bars in your own order?", 'chart.order_x("C", "B", "A")'),
        ],
        task=['Weather table: draw lines of "temperature" over "month", one colour for each city. Put the '
              'legend on top and use the style "paper".'],
        solutions=[(_code('''
            import simplydata as sdata
            weather = sdata.example("weather")
            chart = sdata.chart(weather)
            chart.lines(x="month", y="temperature", colour="city")
            chart.legend_top()
            chart.style("paper")
            chart.save("temperature.png")'''),
            'colour="city" draws one line for each city and adds a legend; the style changes the whole look.')],
    ),
    "families_saving": dict(
        lessons=[],
        manual=[("start:grammar", "How the words are built")],
        words=["save", "hide_legend", "legend_left", "sdata.words"],
        do=['Read "How the words are built" in the manual (two pages). It is the key to guessing words you '
            "have never seen.",
            "Type sdata.words() and look for the families: keep_ and remove_, ..._x and ..._y, add_..._column, "
            "..._of.",
            "Then do the task: guess first, check with sdata.help after."],
        recall=[
            ("If keep_rows keeps rows, what would you guess takes columns away?",
             "remove_columns: verb + thing. keep_ and remove_ work on rows, columns and numbers."),
            ("A word for one axis ends in ...?", "_x or _y: label_y, range_x, log_scale_y."),
            ("Which file endings give a picture that stays sharp at any size?", ".pdf and .svg"),
        ],
        task=["Save one chart as .png, .pdf and .svg. Then guess, before you look anything up: which word hides "
              "the legend, and which puts it on the left? Try your guesses on a chart with colour=."],
        solutions=[(_code('''
            import simplydata as sdata
            marks = sdata.example("marks")
            chart = sdata.chart(marks)
            chart.bars(x="subject", y="marks", colour="class")
            chart.save("subjects.png")
            chart.save("subjects.pdf")
            chart.save("subjects.svg")
            chart.legend_left()
            chart.hide_legend()
            chart.save("no_legend.png")'''),
            "legend_left follows legend_bottom and legend_top; hide_legend follows show_ and hide_. A family "
            "is learned once.")],
    ),
    "lesson4": dict(
        lessons=[4],
        words=["lines", "dots", "histogram", "boxes", "all_dots", "violins"],
        recall=[
            ("Which chart shows how the values of one column are spread, in bins?", 'chart.histogram(x="marks", bins=8)'),
            ("Which two words show the spread of each group, with every value as a dot on top?",
             'chart.boxes(x="class", y="marks") and chart.all_dots(x="class", y="marks")'),
            ("Which word draws one dot for each row, x against y?", 'chart.dots(x="hours", y="marks")'),
            ("Which word draws a line with a dot at each point?", 'chart.lines(x="month", y="temperature", dots=True)'),
        ],
        task=['Draw violins of "temperature" for each city in the weather table, with all the dots on top.'],
        solutions=[(_code('''
            import simplydata as sdata
            weather = sdata.example("weather")
            chart = sdata.chart(weather)
            chart.violins(x="city", y="temperature")
            chart.all_dots(x="city", y="temperature")
            chart.save("violins.png")'''),
            "A violin shows the shape of the spread; the dots show every month.")],
    ),
    "lesson5": dict(
        lessons=[5],
        manual=[("ideas:sd-se", "Standard deviation or standard error?")],
        words=["show=", "error=", "numbers_on_bars", "fit_line", "qq_dots"],
        recall=[
            ("Which option makes a bar show the median instead of the average?", 'show="median"'),
            ("Which option adds error bars that say how sure we are of the average?",
             'error="standard error", or error="confidence interval" for the 95 % interval.'),
            ("Which word draws a straight line through the dots?", "chart.fit_line(), after chart.dots(...)"),
            ("Which word checks by eye whether values are bell-shaped?", 'chart.qq_dots("marks")'),
        ],
        task=['Weather table: draw bars of the average "rain" for each city with 95 % confidence intervals, '
              "and write the numbers on the bars."],
        solutions=[(_code('''
            import simplydata as sdata
            weather = sdata.example("weather")
            chart = sdata.chart(weather)
            chart.bars(x="city", y="rain", error="confidence interval")
            chart.numbers_on_bars()
            chart.save("rain_ci.png")'''),
            "A line is printed saying which error bars were drawn, so the caption can say it too.")],
    ),
    "lesson6": dict(
        lessons=[6],
        words=["range_y", "line_at_y", "shade_between_y", "log_scale_y", "line_at_x", "tick_format_y"],
        recall=[
            ("Which word sets the y-axis from 0 to 100?", "chart.range_y(0, 100)"),
            ("Which word draws a line across at y = 50, with a label?", 'chart.line_at_y(50, label="pass mark")'),
            ("A few values are much bigger than the rest. Which word helps?", "chart.log_scale_y()"),
            ("Which word shows the numbers on the y-axis as percents?", 'chart.tick_format_y("percent")'),
        ],
        task=['Growth table: draw lines of "population" over "year", one for each country, with a line at '
              'x = 2010 and the y-axis starting at 0.'],
        solutions=[(_code('''
            import simplydata as sdata
            growth = sdata.example("growth")
            chart = sdata.chart(growth)
            chart.lines(x="year", y="population", colour="country")
            chart.line_at_x(2010, label="2010")
            chart.range_y(0, 60)
            chart.save("growth.png")'''),
            "range_y takes the lowest and the highest value of the axis; line_at_x draws a line going up.")],
    ),
    "lesson7": dict(
        lessons=[7],
        words=["one_chart_per", "sdata.side_by_side", "sdata.one_above_other", "sdata.grid", "sdata.pairs"],
        recall=[
            ("Which word draws one small chart for each class?", 'chart.one_chart_per("class")'),
            ("Which word puts two charts next to each other on one page?", "sdata.side_by_side(a, b)"),
            ("How do you put four charts in two rows, with the letters A to D?",
             "sdata.grid(a, b, c, d, per_row=2, letters=True)"),
        ],
        task=["Make a histogram and a box chart of the marks, and put them one above the other on one page."],
        solutions=[(_code('''
            import simplydata as sdata
            marks = sdata.example("marks")
            a = sdata.chart(marks).histogram(x="marks")
            b = sdata.chart(marks).boxes(x="class", y="marks")
            page = sdata.one_above_other(a, b)
            page.save("two_charts.png")'''),
            "Each chart is made first; one_above_other puts them on one page.")],
    ),
    "lesson8": dict(
        lessons=[8],
        words=["second_y", "size", "style", "note", "heatmap", "animate"],
        recall=[
            ("Which word adds a second y-axis on the right?", 'chart.second_y("rain")'),
            ("Which word sets the size of a figure in cm?", 'chart.size(width=17, height=8, unit="cm")'),
            ("How do you save a picture sharp enough for a journal?",
             'chart.save("figure.tif", resolution=600), or save it as .pdf'),
            ("Which word writes a note with an arrow on the chart?", 'chart.note("X marks the spot", x=10, y=4, arrow=True)'),
        ],
        task=['A figure for a journal: dots of "hours" against "marks", style "paper", 8.5 cm wide and 7 cm '
              "high, font size 8, saved as a .tif with 600 dpi."],
        solutions=[(_code('''
            import simplydata as sdata
            marks = sdata.example("marks")
            chart = sdata.chart(marks)
            chart.dots(x="hours", y="marks")
            chart.style("paper")
            chart.size(width=8.5, height=7, unit="cm")
            chart.font_size(8)
            chart.save("figure1.tif", resolution=600)'''),
            "Journals often ask for 8.5 cm (one column) or 17 cm (two columns) and 300 to 600 dpi.")],
    ),
    # ------------------------------------------------------------ tables and numbers (lessons 9 to 13)
    "lesson9": dict(
        lessons=[9],
        words=["sdata.table", "table.show_first_rows", "table.describe_columns", "table.count_rows",
               "table.count_each"],
        recall=[
            ("How do you look at the first 5 rows of a table?", "table.show_first_rows(5)"),
            ("Which word describes each column: its kind, empty cells and numbers?", "table.describe_columns()"),
            ("Which word counts the rows of each fruit?", 'table.count_each("fruit")'),
            ("Which word counts the rows?", "table.count_rows()"),
        ],
        task=['Open the "shop" practice table. How many rows does it have, and how many rows does each shop have?'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("shop"))
            table.count_rows()
            table.count_each("shop")
            table.describe_columns()'''),
            "count_rows counts all the rows, count_each the rows of each shop; describe_columns also shows "
            "the empty cells.")],
    ),
    "lesson10": dict(
        lessons=[10],
        words=["table.keep_rows", "table.keep_columns", "table.sort_by", "table.undo", "table.steps",
               "table.keep_rows_by_position"],
        recall=[
            ("Which word keeps only the rows where subject is Maths?", 'table.keep_rows("subject", equal_to="Maths")'),
            ("Which word sorts with the biggest marks first?", 'table.sort_by("marks", biggest_first=True)'),
            ("You made a mistake in the last step. Which word takes it back?",
             "table.undo(); table.start_again() goes back to the start."),
            ("Which word keeps rows 10 to 20, by their place in the table?", "table.keep_rows_by_position(10, to=20)"),
            ("Which word lists every step you did to a table?", "table.steps()"),
        ],
        task=['Weather table: keep the city "Sunnyvale", keep only the columns month, temperature and rain, '
              'sort by rain with the biggest first, and save it as "sunny.csv".'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("weather"))
            table.keep_rows("city", equal_to="Sunnyvale")
            table.keep_columns("month", "temperature", "rain")
            table.sort_by("rain", biggest_first=True)
            table.save("sunny.csv")
            table.steps()'''),
            "Each step prints what it did; steps() lists them all, in order.")],
    ),
    "lesson11": dict(
        lessons=[11],
        manual=[("ideas:missing", "Missing values and imputation")],
        words=["table.count_empty_cells", "table.fill_empty_cells", "table.make_numbers", "table.add_column",
               "table.add_group_column"],
        recall=[
            ("Which word counts the empty cells in each column?", "table.count_empty_cells()"),
            ("How do you fill the empty marks with the median?", 'table.fill_empty_cells("marks", with_the="median")'),
            ('Which word turns a column of text such as "144" into numbers?', 'table.make_numbers("height")'),
            ("Which word adds a new column made from other columns?",
             'table.add_column("total", "Maths + Science"), or table.add_column("per_hour", by_dividing=["marks", "hours"])'),
            ("Which word cuts a column of numbers into named groups?",
             'table.add_group_column("grade", from_column="marks", cut_at=[40, 60, 80], names=["D", "C", "B", "A"])'),
        ],
        task=['Shop table: count the empty cells, fill the empty "sales" with the line between neighbours for '
              'each shop (sorted by "date"), and add a column "weekday" with the name of the weekday.'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("shop"))
            table.count_empty_cells()
            table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date", for_each="shop")
            table.add_date_part_column("weekday", from_column="date", part="weekday name")
            table.show_first_rows()'''),
            "for_each= keeps the two shops apart, so a gap is never filled from the other shop's days.")],
    ),
    "lesson12": dict(
        lessons=[12],
        words=["table.average_of", "table.summary_of", "table.correlation_between", "table.join_with",
               "table.total_of"],
        recall=[
            ("Which word gives the average of a column for each group?", 'table.average_of("marks", for_each="class")'),
            ("Which word tells whether two columns go up and down together?", 'table.correlation_between("hours", "marks")'),
            ("Which word joins two tables on a column they share?", 'table.join_with(ages, using="name")'),
            ("Which word gives several summaries at once?",
             'table.summary_of("sales", for_each="shop", show=["average", "median", "count"])'),
        ],
        task=['Fruit table: the total "count" of each fruit, drawn as bars with the numbers on the bars.'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("fruit"))
            totals = table.total_of("count", for_each="fruit")
            chart = totals.chart()
            chart.bars(x="fruit", y="total of count")
            chart.numbers_on_bars()
            chart.save("fruit_totals.png")'''),
            'A summary gives back a new table, with a column called "total of count", ready for a chart.')],
    ),
    "lesson13": dict(
        lessons=[13],
        words=["sdata.numbers", "sdata.random_numbers", "nums.median", "nums.percentile", "sdata.number_grid"],
        recall=[
            ("How do you make simplydata numbers from a list?", "nums = sdata.numbers([12, 15, 9, 20, 14])"),
            ("Which word gives the middle value of numbers?", "nums.median()"),
            ("How do you make 100 random numbers with average 50 and standard deviation 10, the same every time?",
             "sdata.random_numbers(how_many=100, average=50, standard_deviation=10, seed=1)"),
            ("How do you get the number in row 1, column 2 of a grid?", "grid.number_at(row=1, column=2)"),
        ],
        task=["Make 200 random heights with average 150 and standard deviation 8 (seed=1). What is their 90th "
              "percentile? Draw them as a histogram."],
        solutions=[(_code('''
            import simplydata as sdata
            heights = sdata.random_numbers(how_many=200, average=150, standard_deviation=8, seed=1)
            heights.percentile(90)
            chart = heights.chart()
            chart.histogram(x="value")
            chart.save("heights.png")'''),
            "About 157: nine heights in ten are below it. seed=1 gives the same numbers every time.")],
    ),
    # ------------------------------------------------------------ tests (lessons 14 and 15)
    "tests_a": dict(
        lessons=[14],
        manual=[("ideas:p-value", "The null hypothesis and the p-value"),
                ("ideas:effect-size", "Effect sizes: how big is the difference?")],
        words=["table.compare_averages", "table.compare_each_pair", "table.fit_line"],
        do=[_lesson_do(14, "examples 1 and 3"),
            'Read "The null hypothesis and the p-value" and "Effect sizes" in the manual.'],
        recall=[
            ("Which word tests whether the average marks differ between the classes?",
             'table.compare_averages("marks", for_each="class")'),
            ("Which word says which pairs of groups differ?", 'table.compare_each_pair("marks", for_each="class")'),
            ('A p-value answers "could it be chance?". What answers "how big is it?"',
             "The effect size: the difference in averages, Cohen's d or Hedges' g, omega-squared."),
            ("Which option gives the rank-based test, for small groups that are not bell-shaped?",
             'based_on="ranks"'),
        ],
        task=['School table: do the average marks differ between pupils who had breakfast and pupils who did '
              "not? Write down the difference, the effect size and the p-value."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.average_of("marks", for_each="breakfast")
            table.compare_averages("marks", for_each="breakfast")'''),
            "Two groups, so it is Welch's t-test; the sentence gives the difference, Hedges' g and the p-value.")],
    ),
    "tests_b": dict(
        lessons=[14],
        manual=[("ideas:multiple", "Many tests at once")],
        words=["table.compare_counts", "table.show_rows_with_outliers", "table.compare_shape_to",
               "table.add_adjusted_p_value_column", "table.compare_spreads"],
        do=[_lesson_do(14, "examples 2, 4 and 5"),
            'Read "Many tests at once" in the manual.'],
        recall=[
            ("Which word tests whether the counts of categories differ between groups?",
             'table.compare_counts("result", for_each="class")'),
            ("Which word shows the rows with outliers?", 'table.show_rows_with_outliers("marks", for_each="class")'),
            ("Which word checks whether values are bell-shaped (normal)?", 'table.compare_shape_to("marks", "normal")'),
            ("You did many tests at once. Which word corrects the p-values?",
             'add_adjusted_p_value_column(method="holm"), on the table of results'),
        ],
        task=['School table: is the share of pupils who "passed" different with and without breakfast?'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.count_rows(for_each="breakfast", across="passed")
            table.compare_counts("passed", for_each="breakfast")'''),
            "passed is a category (yes or no), so the test compares counts: the chi-square test, with "
            "Cramer's V as the effect size.")],
    ),
    "choose_test": dict(
        lessons=[],
        manual=[("ideas:which-test", "Which test when?"), ("ideas:report", "Writing it up")],
        words=["table.compare_average_to", "table.compare_before_after", "table.compare_percent_to"],
        do=['Read "Which test when?" and "Writing it up" in the manual. Answer its four questions for each '
            "task below before you run anything.",
            'Run sdata.help("compare_before_after") and sdata.help("compare_average_to") and type their examples.'],
        recall=[
            ("The same children were measured twice. Which word compares the two?",
             'table.compare_before_after("before", "after"): the paired test.'),
            ("Which word compares one average with a number you choose?", 'table.compare_average_to("marks", 60)'),
            ("What should a results sentence give?",
             "The summary of each group with n, the effect size (with its interval), the name of the test and "
             "the exact p-value."),
        ],
        task=["Marks table: is each child's Maths mark different from the same child's Science mark? Choose "
              "the right word first.",
              'School table: is the average "sleep" different from 8 hours?'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("marks"))
            table.keep_columns("name", "subject", "marks")
            table.turn_rows_into_columns(names_from="subject", values_from="marks")
            table.compare_before_after("Maths", "Science")'''),
            "The same children twice, so the test must be paired. First one row per child, with a Maths and "
            "a Science column."),
            (_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.compare_average_to("sleep", 8)'''),
             "One average against a number you choose: the one-sample t-test. Empty sleep cells are skipped, "
             "with a note.")],
    ),
    "lesson15": dict(
        lessons=[15],
        words=["table.steps", "sdata.grid", "table.save", "table.as_pandas"],
        recall=[
            ("Which word lists every change you made to a table, for your methods?", "table.steps()"),
            ("How do you give the panels of a figure the letters A and B?",
             "letters=True in sdata.grid(...) or sdata.side_by_side(...)"),
            ("Which word hands a table on to pandas?", "table.as_pandas()"),
        ],
        task=['From memory: weather table, compare the average "temperature" of the cities, draw boxes with '
              'all the dots, and save the figure as a PDF and the table as "weather.xlsx".'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("weather"))
            table.compare_averages("temperature", for_each="city")
            chart = table.chart()
            chart.boxes(x="city", y="temperature")
            chart.all_dots(x="city", y="temperature")
            chart.save("temperature.pdf")
            table.save("weather.xlsx")'''),
            "Question, test, figure, saved table: the whole analysis in eight lines.")],
    ),
    # ------------------------------------------------------------ going further (lessons 16 to 20)
    "lesson16": dict(
        lessons=[16],
        words=["table.explain", "table.add_predicted_column"],
        recall=[
            ("Which word explains marks from hours, sleep and breakfast together?",
             'table.explain("marks", using=["hours", "sleep", "breakfast"])'),
            ('How do you explain a yes-or-no column such as "passed"?', 'outcome="yes or no" (logistic regression)'),
            ("Which word adds the values a model predicts as a new column?",
             'table.add_predicted_column("marks", using=["hours", "breakfast"])'),
        ],
        task=['School table: explain "days_absent" using "sleep" and "breakfast" with outcome="counts". '
              "Does breakfast matter?"],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.explain("days_absent", using=["sleep", "breakfast"], outcome="counts")'''),
            "Counts of days call for Poisson regression; read the line for breakfast and its p-value.")],
    ),
    "lesson17": dict(
        lessons=[17],
        words=["table.add_principal_components", "table.add_clusters", "table.add_nmds", "table.compare_profiles"],
        recall=[
            ("Which word adds the principal components PC1 and PC2?", "table.add_principal_components(how_many=2)"),
            ("Which word puts rows that are alike into groups?", "table.add_clusters(how_many=2)"),
            ("Which word tests whether groups differ in their make-up across many columns (PERMANOVA)?",
             'table.compare_profiles(for_each="grazed")'),
        ],
        task=['Meadow table: add 3 clusters and count the rows of each cluster across "field".'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("meadow"))
            table.add_clusters(how_many=3)
            table.count_rows(for_each="cluster", across="field")'''),
            "If each cluster is mostly one field, the fields have different plants.")],
    ),
    "lesson18": dict(
        lessons=[18],
        manual=[("ideas:errors", "Two kinds of mistake, and power")],
        words=["table.survival_over_time", "survival_curves", "table.compare_survival", "sdata.how_many_needed"],
        recall=[
            ("Which word draws survival curves?", 'chart.survival_curves(time="days", event="died", colour="treatment")'),
            ("Which word tests whether survival differs between groups?",
             'table.compare_survival(time="days", event="died", for_each="treatment")'),
            ("Which word says how many values you need in each group?", "sdata.how_many_needed(effect_size=0.5, power=0.8)"),
        ],
        task=["How many values does each group need to find a large difference (effect_size=0.8) with power "
              "0.8? And with a small one (0.2)?"],
        solutions=[(_code('''
            import simplydata as sdata
            sdata.how_many_needed(effect_size=0.8, power=0.8)
            sdata.how_many_needed(effect_size=0.2, power=0.8)'''),
            "About 26 for a large difference but about 394 for a small one: small effects need big studies.")],
    ),
    "fitting": dict(
        lessons=[19],
        manual=[("ideas:regression", "Fitting a line or a formula")],
        words=["table.fit_line", "table.fit_curve", "table.fit_formula", "fit_formula"],
        do=[_lesson_do(19, "examples 1 and 2")],
        recall=[
            ("Which word fits a straight line on a table?", 'table.fit_line(x="hours", y="marks")'),
            ("How do you fit a curve with x and x squared?", 'table.fit_curve(x="hours", y="marks", degree=2)'),
            ("Which word fits a formula of your own, such as growth?",
             'table.fit_formula("population = a * exp(b * (year - 2000))")'),
        ],
        task=['Growth table: fit the formula "forest = a + b * (year - 2000)" for each country. Which country '
              "loses forest fastest?"],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("growth"))
            table.fit_formula("forest = a + b * (year - 2000)", for_each="country")'''),
            "b is the change in forest each year: the country with the most negative b loses forest fastest.")],
    ),
    "finding": dict(
        lessons=[19],
        words=["sdata.find_smallest", "sdata.find_biggest", "sdata.solve"],
        do=[_lesson_do(19, "example 3")],
        recall=[
            ("Which word finds where a formula is smallest?", 'sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])'),
            ("Which word finds where the two sides of an equation are equal?", 'sdata.solve("x**3 - 2*x = 5", between=[0, 3])'),
        ],
        task=["A box has a square bottom of side x and a height of 10 - x, so its volume is x * x * (10 - x). "
              "For which x between 0 and 10 is the volume biggest?"],
        solutions=[(_code('''
            import simplydata as sdata
            sdata.find_biggest("x * x * (10 - x)", between=[0, 10])'''),
            "x = 6.67 (that is, 20/3), with a volume of about 148.")],
    ),
    "distributions": dict(
        lessons=[20],
        manual=[("ideas:distribution", "The shape of the values: distributions")],
        words=["sdata.distribution", "dist.chance", "dist.percentile", "table.fit_distribution"],
        do=[_lesson_do(20, "examples 1 and 2")],
        recall=[
            ("How do you make a normal distribution with average 50 and standard deviation 10?",
             'sdata.distribution("normal", average=50, standard_deviation=10)'),
            ("Which word gives the chance of a value above 80?", "dist.chance(above=80)"),
            ("Which word fits a distribution to a column?", 'table.fit_distribution("days_absent", "negative binomial")'),
        ],
        task=["Heights are normal with average 150 and standard deviation 8. What share is taller than 165? "
              "Which height do 95 % of them stay below?"],
        solutions=[(_code('''
            import simplydata as sdata
            heights = sdata.distribution("normal", average=150, standard_deviation=8)
            heights.chance(above=165)
            heights.percentile(95)'''),
            "About 3 % are taller than 165, and 95 % are below about 163.")],
    ),
    "between": dict(
        lessons=[20],
        words=["table.values_at", "table.map_of"],
        do=[_lesson_do(20, "example 3")],
        recall=[
            ("Which word estimates a column between the points you measured?",
             'table.values_at("population", at={"year": [2001, 2011]})'),
            ("Which word fills an even map from scattered points, ready for a contour map?",
             'table.map_of("height", across=["east", "north"])'),
        ],
        task=['Weather table: estimate the "rain" of each city at month_number 2.5 and 7.5, joined by '
              "straight lines."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("weather"))
            rain = table.values_at("rain", at={"month_number": [2.5, 7.5]}, joined_by="straight lines", for_each="city")
            rain.show()'''),
            "Each estimate lies halfway between the two months around it.")],
    ),
    # ------------------------------------------------------------ review days
    "review_p7": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again.",
            "Then do the mixed task without looking back. It uses words from the whole week."],
        task=['Marks table: keep only class "A", draw bars of the average "marks" for each subject with '
              "standard error bars, give the chart a title and save it as a PDF."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("marks"))
            table.keep_rows("class", equal_to="A")
            chart = table.chart()
            chart.bars(x="subject", y="marks", error="standard error")
            chart.title_top("Class A")
            chart.save("class_a.pdf")'''),
            "Day 1 (bars, save), day 4 (error=) and day 6 (keep_rows) in one small program.")],
    ),
    "review_p14": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again.",
            "Then do the mixed task without looking back."],
        task=['School table: count the empty cells, remove the rows with an empty "sleep", give the average '
              'sleep for each value of "passed", test the difference, and draw boxes with all the dots.'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.count_empty_cells()
            table.remove_rows_with_empty_cells("sleep")
            table.average_of("sleep", for_each="passed")
            table.compare_averages("sleep", for_each="passed")
            chart = table.chart()
            chart.boxes(x="passed", y="sleep")
            chart.all_dots(x="passed", y="sleep")
            chart.save("sleep.png")'''),
            "Clean, sum up, test, draw: the order of every analysis.")],
    ),
    "review_p21": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again."],
        task=['Trial table: test whether survival differs between the treatments, then explain the risk of '
              'dying from "treatment" and "age" together.'],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("trial"))
            table.compare_survival(time="days", event="died", for_each="treatment")
            table.explain_survival(time="days", event="died", using=["treatment", "age"])'''),
            "The log-rank test compares the curves; the Cox model gives a hazard ratio for each column.")],
    ),
    "review_p28": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again."],
        task=["Without looking back: which word fits a formula, which finds where it is biggest, and which "
              "gives the chance of a value above a number? Use one of them on any table or formula."],
        solutions=[(_code('''
            import simplydata as sdata
            dist = sdata.distribution("normal", average=50, standard_deviation=10)
            dist.chance(above=70)
            sdata.find_biggest("x * (10 - x)", between=[0, 10])'''),
            "fit_formula, find_biggest and chance: three families (fit_, find_, and the distribution words).")],
    ),
    "review_b7": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again.",
            "Then do the mixed task without looking back."],
        task=["Make numbers of five marks and give their average. Then draw bars of the marks table with "
              'x="subject", y="marks", one colour for each class, and a title. Save it.'],
        solutions=[(_code('''
            import simplydata as sdata
            nums = sdata.numbers([48, 71, 55, 88, 62])
            nums.average()
            marks = sdata.example("marks")
            chart = sdata.chart(marks)
            chart.bars(x="subject", y="marks", colour="class")
            chart.title_top("Marks in each subject")
            chart.save("subjects.png")'''),
            "Python lists (day 2), bars and save (day 3), titles (day 4) and colours (day 5).")],
    ),
    "review_b14": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again.",
            "Then do the mixed task without looking back."],
        task=['Weather table: keep the city "Frostby", sort by "temperature", and draw a line of the temperature '
              "over the month, with a line at y = 0 and the label \"freezing\"."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("weather"))
            table.keep_rows("city", equal_to="Frostby")
            table.sort_by("temperature")
            table.show_first_rows(3)
            chart = table.chart()
            chart.lines(x="month_number", y="temperature")
            chart.line_at_y(0, label="freezing")
            chart.save("frostby.png")'''),
            "Tables (days 12 and 13) feed charts (days 3 to 11).")],
    ),
    "review_b21": dict(
        lessons=[],
        do=["Answer the warm-up on paper, from memory, before you look anything up.",
            "For each question you could not answer, open its day with sdata.day(n) and run its example again.",
            "Then do the mixed task without looking back."],
        task=['School table: the average "marks" for each value of "breakfast", a test of the difference, and '
              "boxes of the marks for each group."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.average_of("marks", for_each="breakfast")
            table.compare_averages("marks", for_each="breakfast")
            chart = table.chart()
            chart.boxes(x="breakfast", y="marks")
            chart.save("breakfast.png")'''),
            "A summary, a test and a chart of the same question.")],
    ),
    # ------------------------------------------------------------ your own project
    "proj_question": dict(
        lessons=[],
        manual=[("ideas:tidy", "Tidy data: long and wide tables")],
        words=["sdata.table", "table.describe_columns", "table.save"],
        do=["Choose data of your own: a file from your work, your studies or a public data set. No data? "
            'Use the "school" practice table, as the answers do.',
            "Write your question in one sentence, for example: do pupils who sleep more get better marks?",
            "Keep the raw file as it is: never change it. All the cleaning will be steps on a table, which "
            "you can repeat and undo.",
            "Open the file with sdata.table and look at it."],
        task=["Write your question in one sentence. Open your raw file and write down how many rows and "
              "columns it has, and which columns have empty cells."],
        solutions=[(_code('''
            import simplydata as sdata
            sdata.table(sdata.example("school")).save("school_raw.csv")
            table = sdata.table("school_raw.csv")
            table.count_rows()
            table.describe_columns()'''),
            "The first line only makes a raw file to practise on; with your own data, start at the second "
            "line. The school table has 180 rows and 8 columns; sleep has 19 empty cells.")],
    ),
    "proj_clean": dict(
        lessons=[11],
        words=["table.count_empty_cells", "table.remove_extra_spaces", "table.replace_values",
               "table.fill_empty_cells", "table.rename_column"],
        do=["Project day 2: clean. Go column by column through what describe_columns showed you.",
            "Fix text with spaces or mixed capitals, numbers stored as text, odd codes such as -999, and "
            "decide what to do with empty cells (remove, fill, or keep and let the tests skip them).",
            "Save the clean table under a new name, and keep the steps()."],
        task=["Clean your table and save it as a new file. Write one line for each decision you made about "
              "empty cells."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.count_empty_cells()
            table.add_is_empty_column("sleep_was_empty", from_column="sleep")
            table.rename_column("days_absent", to="absent")
            table.save("school_clean.csv")
            table.steps()'''),
            "Here the empty sleep cells are kept (tests skip them, and models can use multiple imputation); "
            "a column remembers which were empty.")],
    ),
    "proj_explore": dict(
        lessons=[9, 12],
        words=["table.count_each", "table.summary_of", "table.correlation_between", "histogram", "sdata.pairs"],
        do=["Project day 3: explore. Draw before you test: charts show odd values, skew and outliers that a "
            "single number hides.",
            "Count each category, sum up the numbers for each group, look at histograms and at sdata.pairs."],
        task=["Make three summaries and two charts of your table that help to answer your question."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.count_each("breakfast")
            table.summary_of("marks", for_each="breakfast", show=["average", "median", "standard deviation", "count"])
            table.correlation_between("sleep", "marks")
            chart = table.chart()
            chart.histogram(x="marks", colour="breakfast")
            chart.save("marks_histogram.png")
            sdata.pairs(table, colour="breakfast").save("pairs.png")'''),
            "Each summary and chart is a small answer; the test comes later.")],
    ),
    "proj_figures": dict(
        lessons=[7, 8, 15],
        words=["sdata.grid", "style", "size", "save"],
        do=["Project day 4: figures. Make the one figure that answers your question best, with two panels "
            "(A and B), axis names with units, and a caption that says what the error bars or bands show.",
            'Use style "paper", a size of 17 cm wide, and save it as .pdf and as a 600 dpi .tif.'],
        task=["Make your main figure with two panels and save it ready for a journal."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            a = table.chart()
            a.boxes(x="breakfast", y="marks")
            a.all_dots(x="breakfast", y="marks")
            a.label_y("Mark (out of 100)")
            b = table.chart()
            b.dots(x="sleep", y="marks", colour="breakfast")
            b.fit_line(confidence_band=True)
            b.label_x("Sleep (hours)")
            figure = sdata.grid(a, b, per_row=2, letters=True)
            figure.style("paper")
            figure.size(width=17, height=8, unit="cm")
            figure.save("figure1.pdf")
            figure.save("figure1.tif", resolution=600)'''),
            "Panel A shows the groups, panel B the link with sleep; the band is the 95 % confidence interval.")],
    ),
    "proj_tests": dict(
        lessons=[14],
        manual=[("ideas:which-test", "Which test when?")],
        words=["table.compare_averages", "table.compare_counts", "table.correlation_between",
               "table.compare_shape_to"],
        do=["Project day 5: tests. Choose the test with the four questions of \"Which test when?\" before "
            "you run it, and run one test per question.",
            "Check the shape and the outliers first; use based_on=\"ranks\" if the groups are small and far "
            "from bell-shaped."],
        task=["Run the test that answers your question. Write down the test's name, the effect size and the p-value."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.compare_shape_to("marks", "normal", for_each="breakfast")
            table.compare_averages("marks", for_each="breakfast")
            table.correlation_between("sleep", "marks")'''),
            "Two groups of numbers: Welch's t-test. Two columns of numbers: Pearson's correlation.")],
    ),
    "proj_model": dict(
        lessons=[16, 19],
        words=["table.explain", "table.fit_line", "table.add_predicted_column"],
        do=["Project day 6: go further. If several columns could explain the answer, put them in one model "
            "with explain(); if one column follows another, fit a line or a formula.",
            "Compare the model with the simple test of yesterday: does the answer hold when the other "
            "columns are taken into account?"],
        task=["Explain your main column from two or three others, and say in one sentence what changed "
              "compared with the single test."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.explain("marks", using=["hours", "sleep", "breakfast"])
            table.fit_line(x="sleep", y="marks")'''),
            "Each line of the model is the effect of one column with the others held the same.")],
    ),
    "proj_text": dict(
        lessons=[],
        manual=[("ideas:report", "Writing it up")],
        words=["table.steps"],
        do=["Project day 7: write. Turn the printed answers into a results paragraph: for each finding, the "
            "summary of each group with n, the effect size, the name of the test and the exact p-value.",
            "Write the methods from table.steps(): what you removed or filled, and which tests you used.",
            "Write the caption of your figure: what each panel shows, what the bars or bands are, and n."],
        task=["Write a results paragraph, a methods paragraph and a figure caption for your project."],
        solutions=[("", 'For example: "Pupils who had breakfast scored on average 3.4 marks more than pupils '
                        "who did not (49.7 against 46.3; n = 115 and 65), but a difference this big could be "
                        "chance (Welch's t-test, p = 0.051; Hedges' g = 0.31). Sleep was not recorded for 19 "
                        'pupils; they were left out of the tests that use sleep." Use your own numbers, '
                        "exactly as printed, and report a result that could be chance as plainly as one that "
                        "could not.")],
    ),
    "proj_finish": dict(
        lessons=[],
        words=["table.steps", "table.save"],
        do=["Project day 8: finish. Put every step of your project in one script, from the raw file to the "
            "figure, and run it from the top in a fresh session. It must run without errors.",
            "Read steps() out loud: would someone else understand what you did?"],
        task=["Run your whole project script from the top. Check that it makes the clean table, the figure "
              "and the numbers in your text."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.add_is_empty_column("sleep_was_empty", from_column="sleep")
            table.compare_averages("marks", for_each="breakfast")
            chart = table.chart()
            chart.boxes(x="breakfast", y="marks")
            chart.save("figure1.pdf")
            table.save("school_clean.csv")
            table.steps()'''),
            "One script from the raw file to the figure: anyone can run it again and get the same answers.")],
    ),
    "bproj_a": dict(
        lessons=[9, 11, 12],
        words=["sdata.table", "table.describe_columns", "table.count_empty_cells", "table.summary_of"],
        do=["Your small project, day 1. Choose data of your own (a file from school, work or a public data "
            'set), or use the "school" practice table.',
            "Write your question in one sentence. Open the file, look at it, clean what needs cleaning and "
            "make two summaries."],
        task=["Write your question. Open your file, describe its columns, deal with the empty cells and make "
              "two summaries for each group."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.describe_columns()
            table.count_empty_cells()
            table.remove_rows_with_empty_cells("sleep")
            table.average_of("marks", for_each="breakfast")
            table.summary_of("sleep", for_each="breakfast", show=["average", "count"])'''),
            "Question: do pupils who eat breakfast get better marks? Look, clean, sum up.")],
    ),
    "bproj_b": dict(
        lessons=[14, 15],
        manual=[("ideas:report", "Writing it up")],
        words=["table.compare_averages", "boxes", "save"],
        do=["Your small project, day 2. Test the difference, make one clear figure, and write two sentences "
            "with the numbers as they were printed."],
        task=["Test your question, draw one figure and save it, and write two sentences of results."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("school"))
            table.compare_averages("marks", for_each="breakfast")
            chart = table.chart()
            chart.boxes(x="breakfast", y="marks")
            chart.all_dots(x="breakfast", y="marks")
            chart.label_y("Mark (out of 100)")
            chart.save("breakfast.png")'''),
            "Two sentences: the averages of both groups, then the test's name, the difference and the p-value.")],
    ),
    # ------------------------------------------------------------ the last two days
    "final_test": dict(
        lessons=[],
        words=["sdata.words", "sdata.help"],
        do=["Today's warm-up is the final test: answer every question on paper, from memory. Then check "
            "your answers with sdata.answers().",
            "Then write the whole program of the task in an empty file, without looking anything up. Only "
            "when it is finished, check it with sdata.help."],
        task=['From memory, in an empty file: open the "marks" table, remove the rows with empty cells, give '
              'the average marks for each class, test the difference, and save a two-panel figure (boxes, '
              "and dots of hours against marks with a fitted line) as a PDF."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("marks"))
            table.remove_rows_with_empty_cells()
            table.average_of("marks", for_each="class")
            table.compare_averages("marks", for_each="class")
            a = table.chart()
            a.boxes(x="class", y="marks")
            b = table.chart()
            b.dots(x="hours", y="marks")
            b.fit_line()
            figure = sdata.grid(a, b, per_row=2, letters=True)
            figure.save("final.pdf")'''),
            "If you wrote this from memory, you know the core of simplydata. Words you missed: run their "
            "sdata.help again tomorrow.")],
    ),
    "compare_pandas": dict(
        lessons=[15],
        words=["table.as_pandas", "table.as_polars", "table.steps"],
        do=["Today: the same work in pandas and matplotlib, the libraries simplydata uses underneath. Each "
            "word of the manual's dictionary says which one does the work (\"Underneath\").",
            "Run the simplydata version and the pandas version of the task. Compare them line by line.",
            "Look back at the calendar with sdata.day() and choose three days to do again next month."],
        task=["In simplydata: keep classes A and B of the marks table, give the average marks of each class, "
              "test the difference and draw it as bars.",
              "Write the same analysis with pandas, SciPy and matplotlib, then compare the two line by line."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("marks"))
            table.keep_rows("class", one_of=["A", "B"])
            table.average_of("marks", for_each="class")
            table.compare_averages("marks", for_each="class")
            chart = table.chart()
            chart.bars(x="class", y="marks")
            chart.save("simplydata_bars.png")'''),
            "The simplydata version: one word per step, and each answer printed in a sentence."),
            (_code('''
            import matplotlib.pyplot as plt
            import simplydata as sdata
            from scipy import stats
            df = sdata.example("marks")
            df = df[df["class"].isin(["A", "B"])]
            print(df.groupby("class")["marks"].mean())
            a = df.loc[df["class"] == "A", "marks"]
            b = df.loc[df["class"] == "B", "marks"]
            print(stats.ttest_ind(a, b, equal_var=False))
            fig, ax = plt.subplots()
            means = df.groupby("class")["marks"].mean()
            ax.bar(means.index, means.values)
            fig.savefig("pandas_bars.png")'''),
             "The same in pandas, SciPy and matplotlib: equal_var=False makes it Welch's t-test, as in "
             "simplydata.")],
    ),
    "look_back": dict(
        lessons=[],
        words=["sdata.day", "sdata.words", "table.as_pandas"],
        do=["Look back over the 30 days with sdata.day(): it lists every day.",
            "Choose the three days you found hardest and do their tasks again, without looking back.",
            "Take a first look at pandas, the library underneath simplydata tables: table.as_pandas() hands "
            "your table over, and the manual's dictionary says which library does the work of each word."],
        task=["Do the task of your hardest day again from memory. Then hand the marks table to pandas and "
              "print its first rows with pandas' own word head()."],
        solutions=[(_code('''
            import simplydata as sdata
            table = sdata.table(sdata.example("marks"))
            df = table.as_pandas()
            print(df.head())'''),
            "head() is pandas' word for show_first_rows(). simplydata knows many such names: "
            'sdata.help("head") points to the simplydata word.')],
    ),
}

# ---------------------------------------------------------------- the two tracks
# (title, goal, units, kind). kind: "learn", "review", "project" or "final".
_PYTHON = [
    ("Your first chart and your first table",
     "Make a bar chart with titles and labels, and a first table you keep rows of and sum up.",
     ["lesson1", "lesson2", "first_table"], "learn"),
    ("Colours, word families and saving",
     "Give groups their own colours, learn how simplydata words are built, and save in every file type.",
     ["lesson3", "families_saving"], "learn"),
    ("Every kind of chart",
     "Draw lines, dots, histograms, boxes, violins, pies and areas, and know when to use each.",
     ["lesson4"], "learn"),
    ("Statistics on charts, and axes",
     "Show averages with error bars and fit lines, and control the axes with ranges and reference lines.",
     ["lesson5", "lesson6"], "learn"),
    ("Several charts, and a figure for a journal",
     "Put charts side by side or in a grid, and save a figure at the size and resolution a journal asks for.",
     ["lesson7", "lesson8"], "learn"),
    ("Open a table, choose rows and columns",
     "Look at a table, keep and remove rows and columns (by value and by position), sort, and undo.",
     ["lesson9", "lesson10"], "learn"),
    ("Review of the first week",
     "Remember the words of days 1 to 6 without looking, and use them together in one program.",
     ["review_p7"], "review"),
    ("Clean the data and add columns",
     "Find and fill empty cells, fix text and numbers, and add new columns, groups and dates.",
     ["lesson11"], "learn"),
    ("Summaries, numbers and grids",
     "Sum up columns for each group, join and reshape tables, and work with lists and grids of numbers.",
     ["lesson12", "lesson13"], "learn"),
    ("Is a difference real? Tests and effect sizes",
     "Test whether averages differ, find which groups differ, and say how big the difference is.",
     ["tests_a"], "learn"),
    ("Counts, outliers and many tests",
     "Test counts of categories, check shapes and outliers, and correct the p-values of many tests.",
     ["tests_b"], "learn"),
    ("Choosing a test, and writing it up",
     "Choose the right test with four questions, and write a results sentence a reader can check.",
     ["choose_test"], "learn"),
    ("A whole analysis",
     "Go from a question and a file to a tested answer and a figure with panels A and B.",
     ["lesson15"], "learn"),
    ("Review of the second week",
     "Remember the table and test words of days 6 to 13 without looking, and use them together.",
     ["review_p14"], "review"),
    ("Models: explain one column from several",
     "Explain one column from several others: multiple, logistic, Poisson and mixed regression.",
     ["lesson16"], "learn"),
    ("Patterns across many columns",
     "Find patterns across many columns with principal components, clusters, NMDS and PERMANOVA.",
     ["lesson17"], "learn"),
    ("Survival, and planning a study",
     "Draw and test survival curves, and work out how many values a study needs.",
     ["lesson18"], "learn"),
    ("Fitting lines, curves and formulas",
     "Fit straight lines, curves and formulas of your own to a table, and draw them on a chart.",
     ["fitting"], "learn"),
    ("Finding, solving and distributions",
     "Find where a formula is smallest or biggest, solve equations, and work out chances from a distribution.",
     ["finding", "distributions"], "learn"),
    ("Values between points",
     "Estimate values between the points you measured, and fill an even map for a contour chart.",
     ["between"], "learn"),
    ("Review, and the question of your project",
     "Remember the words of the third week, then choose the question and the raw file of your own project.",
     ["review_p21", "proj_question"], "review"),
    ("Project: clean",
     "Clean your raw file into a tidy table, with every step written down.",
     ["proj_clean"], "project"),
    ("Project: explore",
     "Explore your clean table with summaries and charts before any test.",
     ["proj_explore"], "project"),
    ("Project: figures",
     "Make the main figure of your project, ready for a journal.",
     ["proj_figures"], "project"),
    ("Project: tests",
     "Choose and run the test that answers your question, with its effect size.",
     ["proj_tests"], "project"),
    ("Project: models and fits",
     "Check your answer with a model of several columns, or a fitted line or formula.",
     ["proj_model"], "project"),
    ("Project: results text",
     "Write the results, the methods and the figure caption of your project from the printed answers.",
     ["proj_text"], "project"),
    ("Review, and finish your project",
     "Remember words from the whole course, and run your project as one script from the raw file to the figure.",
     ["review_p28", "proj_finish"], "review"),
    ("Final test from memory",
     "Answer questions on the whole course and write a whole analysis from memory.",
     ["final_test"], "final"),
    ("The same work in pandas and matplotlib",
     "See what simplydata does underneath by writing the same analysis in pandas, SciPy and matplotlib.",
     ["compare_pandas"], "learn"),
]

_BEGINNER = [
    ("Python first steps: values and names",
     "Run your first lines of Python: text in quotes, numbers, names for values, and print.",
     ["py_values"], "learn"),
    ("Python first steps: lists and words",
     "Make lists, call a word with brackets and options, and use simplydata for the first time.",
     ["py_lists"], "learn"),
    ("Reading an error, and your first chart",
     "Read an error message calmly from the bottom up, then make and save your first chart.",
     ["py_errors", "lesson1"], "learn"),
    ("Titles and labels",
     "Put a title, axis names and a caption on a chart, and make its text easy to read.",
     ["lesson2"], "learn"),
    ("Colours, groups and legends",
     "Give each group its own colour and place the legend where it helps.",
     ["lesson3"], "learn"),
    ("Different chart types",
     "Draw lines, dots, histograms, boxes, violins, pies and areas.",
     ["lesson4"], "learn"),
    ("Review of the first week",
     "Remember the Python and chart words of days 1 to 6 without looking, and use them together.",
     ["review_b7"], "review"),
    ("Averages, error bars and fit lines",
     "Show averages and medians with error bars, and draw a line through dots.",
     ["lesson5"], "learn"),
    ("Axes and reference lines",
     "Choose the range of an axis, and add lines and shading at the values that matter.",
     ["lesson6"], "learn"),
    ("Several charts together",
     "Put several charts on one page: one chart per group, side by side, or in a grid.",
     ["lesson7"], "learn"),
    ("A figure for a journal, and word families",
     "Size and save a figure for a journal, and learn the patterns that make words easy to guess.",
     ["lesson8", "families_saving"], "learn"),
    ("Open a table and look at it",
     "Open a table, look at its first rows and columns, and count its rows.",
     ["lesson9"], "learn"),
    ("Choose rows and columns, and undo",
     "Keep and remove rows and columns, sort, and take back any step.",
     ["lesson10"], "learn"),
    ("Review of the second week",
     "Remember the words of days 8 to 13 without looking, and use them together.",
     ["review_b14"], "review"),
    ("Clean the data and add columns",
     "Find and fill empty cells, fix text and numbers, and add new columns.",
     ["lesson11"], "learn"),
    ("Summaries, and from a table to a chart",
     "Sum up columns for each group, and draw a summary as a chart.",
     ["lesson12"], "learn"),
    ("Lists and grids of numbers",
     "Do maths on lists and grids of numbers, and make random numbers.",
     ["lesson13"], "learn"),
    ("Is a difference real? Tests and effect sizes",
     "Test whether averages differ, and say how big the difference is.",
     ["tests_a"], "learn"),
    ("Counts, outliers and many tests",
     "Test counts of categories, and check shapes and outliers.",
     ["tests_b"], "learn"),
    ("Choosing a test, and a whole analysis",
     "Choose the right test, and go from a question to a tested answer and a figure.",
     ["choose_test", "lesson15"], "learn"),
    ("Review of the third week",
     "Remember the words of days 15 to 20 without looking, and use them together.",
     ["review_b21"], "review"),
    ("Models: explain one column from several",
     "Explain one column from several others with a model.",
     ["lesson16"], "learn"),
    ("Patterns across many columns",
     "Find patterns across many columns with principal components and clusters.",
     ["lesson17"], "learn"),
    ("Survival, and planning a study",
     "Draw survival curves, and work out how many values a study needs.",
     ["lesson18"], "learn"),
    ("Fitting, finding and solving",
     "Fit lines, curves and formulas, find where a formula is biggest, and solve equations.",
     ["fitting", "finding"], "learn"),
    ("Distributions and values between points",
     "Work out chances from a distribution, and estimate values between measured points.",
     ["distributions", "between"], "learn"),
    ("Small project: look, clean and sum up",
     "Choose a question and a file, then look at it, clean it and sum it up.",
     ["bproj_a"], "project"),
    ("Small project: test, figure and results",
     "Answer your question with a test, a figure and two sentences of results.",
     ["bproj_b"], "project"),
    ("Final test from memory",
     "Answer questions on the whole course and write a whole analysis from memory.",
     ["final_test"], "final"),
    ("Look back, and what comes next",
     "Do your hardest day again, and take a first look at pandas underneath simplydata.",
     ["look_back"], "learn"),
]

DAYS = {"python": _PYTHON, "beginner": _BEGINNER}


# ---------------------------------------------------------------- putting a day together
def _unit_do(name: str) -> list[str]:
    unit = UNITS[name]
    if "do" in unit:
        return list(unit["do"])
    return [_lesson_do(n) for n in unit["lessons"]]


def _has_recall(track: str, number: int) -> bool:
    return any(UNITS[u].get("recall") for u in DAYS[track][number - 1][2])


def _recall_of(track: str, number: int) -> list[tuple[str, str]]:
    return [qa for u in DAYS[track][number - 1][2] for qa in UNITS[u].get("recall", [])]


# How far back the warm-up looks: yesterday, about three days ago, about a week ago and, later,
# two weeks ago. Review and final days look back over everything.
_GAPS = (1, 3, 7, 14)
_REVIEW_GAPS = (1, 2, 3, 4, 5, 6, 7, 10, 14, 21)
_FINAL_GAPS = (1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 25, 28)


def _source_days(track: str, number: int) -> list[int]:
    """The earlier days the warm-up of a day asks about: for each gap, the nearest earlier day with
    recall questions that is not already chosen."""
    kind = DAYS[track][number - 1][3]
    gaps = {"review": _REVIEW_GAPS, "final": _FINAL_GAPS}.get(kind, _GAPS)
    chosen: list[int] = []
    for gap in gaps:
        day = number - gap
        while day >= 1 and (day in chosen or not _has_recall(track, day)):
            day -= 1
        if day >= 1:
            chosen.append(day)
    return chosen


MIN_WARM_UP = 3


def _warm_up_sources(track: str, number: int, last_asked: dict[int, int], used: dict[int, int]) -> list[int]:
    """The source day of each warm-up question.

    Learning days follow the gaps: yesterday, about three days ago, about a week ago. Early on, when
    there are fewer earlier days than questions, the most recent days are asked about twice (with a
    different question). Review, project and final days mix the whole course: first the days with
    questions not asked yet, then the days asked about longest ago."""
    sources = _source_days(track, number)
    if DAYS[track][number - 1][3] != "learn":
        earlier = [d for d in range(1, number) if _has_recall(track, d)]
        unasked = {d: used.get(d, 0) < len(_recall_of(track, d)) for d in earlier}
        earlier.sort(key=lambda d: (not unasked[d], last_asked.get(d, 0), -d))
        return earlier[:max(len(sources), MIN_WARM_UP)]
    earlier = [d for d in range(number - 1, 0, -1) if _has_recall(track, d)]
    i = 0
    while earlier and len(sources) < MIN_WARM_UP:
        sources.append(earlier[i % len(earlier)])
        i += 1
    return sources


def _warm_up_table(track: str) -> list[list[tuple[str, str, int]]]:
    """The warm-up of every day: [(question, answer, from day)]. Each time a day is asked about, the
    next of its questions is used, so the questions change."""
    used: dict[int, int] = {}
    last_asked: dict[int, int] = {}
    out = []
    for number in range(1, DAYS_IN_A_TRACK + 1):
        found = []
        for source in _warm_up_sources(track, number, last_asked, used):
            last_asked[source] = number
            questions = _recall_of(track, source)
            q, a = questions[used.get(source, 0) % len(questions)]
            used[source] = used.get(source, 0) + 1
            found.append((q, a, source))
        out.append(found)
    return out


_WARM_UPS: dict[str, list] = {}


def warm_up(track: str, number: int) -> list[tuple[str, str, int]]:
    """The warm-up questions of a day: [(question, answer, the day it comes from)]."""
    if track not in _WARM_UPS:
        _WARM_UPS[track] = _warm_up_table(track)
    return _WARM_UPS[track][number - 1]


WORDS_PER_UNIT = 4  # the words to look up on a day: the first ones of each unit, at most 8
WORDS_PER_DAY = 8


def plan(track: str, number: int) -> dict:
    """Everything about one day, put together from its units."""
    title, goal, units, kind = DAYS[track][number - 1]
    found = dict(track=track, number=number, title=title, goal=goal, kind=kind, units=list(units),
                 lessons=[], manual=[], words=[], do=[], examples=[], errors=[], task=[], solutions=[],
                 warm_up=warm_up(track, number))
    for name in units:
        unit = UNITS[name]
        for n in unit["lessons"]:
            if n not in found["lessons"]:
                found["lessons"].append(n)
        for ref in unit.get("manual", []):
            if ref not in found["manual"]:
                found["manual"].append(ref)
        for word in unit.get("words", [])[:WORDS_PER_UNIT]:
            if word not in found["words"] and len(found["words"]) < WORDS_PER_DAY:
                found["words"].append(word)
        found["do"] += _unit_do(name)
        found["examples"] += unit.get("examples", [])
        found["errors"] += unit.get("errors", [])
        found["task"] += unit.get("task", [])
        found["solutions"] += unit.get("solutions", [])
    return found


def all_code() -> list[tuple[str, str]]:
    """(where, code) for every piece of code in the 30 days that must run: examples and solutions."""
    found = []
    for name, unit in UNITS.items():
        for i, block in enumerate(unit.get("examples", []), start=1):
            body = "\n".join(c for c, _ in block)
            found.append((f"{name} example {i}", body))
        for i, (source, _) in enumerate(unit.get("solutions", []), start=1):
            if source:
                found.append((f"{name} solution {i}", source))
    return found


# ---------------------------------------------------------------- progress: one small file on this computer
# Only three things are kept: the track, the day you started and the date each day was first opened.
# Nothing is ever sent anywhere. SIMPLYDATA_PROGRESS_FILE chooses another file (the tests use it).
PROGRESS_ENV = "SIMPLYDATA_PROGRESS_FILE"
GUIDE_URL = "https://github.com/bioforkgithub/simplydata/blob/main/docs/manual/simplydata_30_days.pdf"
MANUAL_NAME = "simplydata_manual.pdf"


def progress_file() -> Path:
    """Where the progress is kept: SIMPLYDATA_PROGRESS_FILE, else .simplydata/progress.json in your home folder."""
    chosen = os.environ.get(PROGRESS_ENV)
    return Path(chosen).expanduser() if chosen else Path.home() / ".simplydata" / "progress.json"


def _date_today() -> datetime.date:
    return datetime.date.today()


def _read_progress() -> dict | None:
    path = progress_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        progress = dict(track=data["track"], started=datetime.date.fromisoformat(data["started"]),
                        opened={int(n): datetime.date.fromisoformat(d) for n, d in data.get("opened", {}).items()})
        if progress["track"] not in TRACK_NAMES or not all(1 <= n <= DAYS_IN_A_TRACK for n in progress["opened"]):
            raise ValueError("unknown track or day")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        print(_wrap(f"(Your progress file {path} could not be read, so this is shown as if you had not started. "
                    "sdata.start_30_days(start_again=True) makes a new one.)"))
        print()
        return None
    return progress


def _write_progress(progress: dict) -> None:
    path = progress_file()
    data = {
        "about": "Progress in 30 days of simplydata, kept only on this computer. Delete this file to forget it.",
        "track": progress["track"],
        "started": progress["started"].isoformat(),
        "opened": {str(n): d.isoformat() for n, d in sorted(progress["opened"].items())},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        print(_wrap(f"(Your progress could not be saved in {path}. To keep it in another file, set the "
                    f"environment variable {PROGRESS_ENV} to that file's name.)"))


# ---------------------------------------------------------------- checking what was typed
def _track(track) -> str:
    """"python" or "beginner", from what was typed ("Python", "beginners" ...)."""
    typed = track.strip().lower() if isinstance(track, str) else None
    if typed is not None and typed.rstrip("s") in TRACK_NAMES:
        return typed.rstrip("s")
    lines = [f"There is no track called {track!r}. There are two:",
             '  track="python"    for people who already write some Python',
             '  track="beginner"  for people who have never programmed']
    guesses = suggest(typed, list(TRACK_NAMES), n=1) if typed else []
    if guesses:
        lines.insert(1, f'Did you mean track="{guesses[0]}"?')
    raise SimplydataError("\n".join(lines))


def _day_number(number, word: str) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= DAYS_IN_A_TRACK:
        raise SimplydataError(
            f"There are days 1 to {DAYS_IN_A_TRACK}, so sdata.{word}({number!r}) does not exist.\n"
            f"Write the day as a whole number, such as sdata.{word}(5); sdata.day() lists all {DAYS_IN_A_TRACK} days.")
    return number


# ---------------------------------------------------------------- printing a day
WIDTH = 78


def _wrap(text: str, indent: str = "", first: str | None = None) -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent if first is None else first, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def help_name(key: str) -> str:
    """How a word is written in sdata.help("..."): "table.keep_rows", "bars", "show=", "today"."""
    return key[len("sdata."):] if key.startswith("sdata.") else key


def _track_title(track: str) -> str:
    return TRACKS[track]["title"]


def _print_day(p: dict, footer: str) -> None:
    number, track = p["number"], p["track"]
    title = f"Day {number} of {DAYS_IN_A_TRACK} ({_track_title(track)}): {p['title']}"
    print(title)
    print("=" * min(len(title), WIDTH))
    print(_wrap("Goal: " + p["goal"] + " About an hour."))
    print()
    print("1. Warm-up, 5 minutes: answer from memory, without looking back")
    if p["warm_up"]:
        for i, (question, _, source) in enumerate(p["warm_up"], start=1):
            print(_wrap(f"(day {source}) {question}", "       ", f"   {i:>2}. "))
        print(f"   Then check: sdata.answers({number})")
    else:
        print("   Nothing to remember yet on day 1: just start.")
    print()
    print("2. What to do")
    for line in p["do"]:
        print(_wrap(line, "     ", "   - "))
    if p["manual"] or p["lessons"]:
        refs = [f"lesson {n}" for n in p["lessons"]] + [f'"{t}"' for _, t in p["manual"]]
        print(_wrap(f"In the manual ({MANUAL_NAME}): " + ", ".join(refs) + ".", "     ", "   "))
    if p["words"]:
        print(_wrap("Words to look up: " + ", ".join(f'sdata.help("{help_name(k)}")' for k in p["words"]),
                    "     ", "   "))
    for block in p["examples"]:
        print()
        print("   Example (type it yourself)")
        for code, _ in block:
            print(f"       {code}")
        print("   What each line does")
        for code, explain in block:
            print(f"       {code}")
            print(_wrap(explain, "           "))
    if p["errors"]:
        print()
        print("   Errors to make on purpose (read the last line of each error first)")
        for code, _, meaning in p["errors"]:
            for line in code.splitlines():
                print(f"       {line}")
            print(_wrap(meaning, "           "))
    print()
    print("3. Try it yourself, without looking back")
    for i, task in enumerate(p["task"], start=1):
        print(_wrap(task, "       ", f"   {i:>2}. "))
    print(f"   Stuck, or done? Worked answers: sdata.answers({number})")
    print()
    print(_wrap(footer))


def _print_answers(p: dict) -> None:
    number = p["number"]
    title = f"Answers to day {number} ({_track_title(p['track'])}): {p['title']}"
    print(title)
    print("=" * min(len(title), WIDTH))
    if p["warm_up"]:
        print()
        print("Warm-up")
        for i, (question, answer, source) in enumerate(p["warm_up"], start=1):
            print(_wrap(f"(day {source}) {question}", "       ", f"   {i:>2}. "))
            print(_wrap(answer, "       ", "       "))
    print()
    print("Try it yourself")
    print(_wrap("There is often more than one good answer: if yours runs and answers the question, it is right.",
                "   ", "   "))
    for i, (task, (code, explain)) in enumerate(zip(p["task"], p["solutions"]), start=1):
        print()
        print(_wrap(task, "       ", f"   {i:>2}. "))
        if code:
            print(textwrap.indent(code, "         "))
        print(_wrap(explain, "       ", "       "))


def _next_footer(track: str, number: int, browsing: bool) -> str:
    if number >= DAYS_IN_A_TRACK:
        return ("This is the last day. Well done for getting here! Go back to any day with sdata.day(n), "
                "and keep using sdata.help() as you work.")
    title = DAYS[track][number][0]
    if browsing:
        return f"Next: sdata.day({number + 1})  {title}"
    return (f"Tomorrow: day {number + 1}, {title}. Type sdata.today() tomorrow, or sdata.next_day() to go on "
            "now.")


def _print_not_started() -> None:
    print(_wrap("You have not started the 30 days of simplydata yet. Choose a track and start with one line:"))
    print("    sdata.start_30_days()                    if you already write some Python")
    print('    sdata.start_30_days(track="beginner")    if you have never programmed')
    print(_wrap("Below is day 1 of the Python track, so you can see what a day looks like. Nothing is "
                "marked as started."))
    print()


def _print_finished(progress: dict) -> None:
    track = progress["track"]
    print(_wrap(f"You have opened all {DAYS_IN_A_TRACK} days of the {_track_title(track)}. Well done!"))
    print(_wrap("Look back at any day with sdata.day(n) and its answers with sdata.answers(n). To go "
                f'through them again from day 1, type sdata.start_30_days(track="{track}", start_again=True).'))


def _open_day(progress: dict, number: int, notes: list[str]) -> None:
    """Remember that a day was opened today, then show it."""
    if number not in progress["opened"]:
        progress["opened"][number] = _date_today()
        _write_progress(progress)
    for note in notes:
        print(_wrap(note))
        print()
    _print_day(plan(progress["track"], number), _next_footer(progress["track"], number, browsing=False))


def _when(date: datetime.date) -> str:
    return f"{date.day} {date:%B %Y}"


# ---------------------------------------------------------------- the words
def start_30_days(track="python", start_again=False) -> None:
    """Start 30 days of simplydata: an hour a day of warm-up questions, lessons and tasks. sdata.start_30_days()

    track="python" (the normal choice) is for people who already write some Python; track="beginner"
    starts with three days of just enough Python. The day you start is kept in a small file on this
    computer (nothing is sent anywhere), so that sdata.today() knows which day comes next. If you have
    started already, it says so and changes nothing; start_again=True starts again from day 1:
    sdata.start_30_days(track="beginner", start_again=True).
    """
    track = _track(track)
    old = _read_progress()
    if old is not None and not start_again:
        print(_wrap(f"You started the 30 days ({_track_title(old['track'])}) on {_when(old['started'])} and have "
                    f"opened {len(old['opened'])} of the {DAYS_IN_A_TRACK} days. Type sdata.today() to carry on."))
        again = old["track"]
        other = next(t for t in TRACK_NAMES if t != again)
        print(_wrap(f'To start the {_track_title(again)} again from day 1, type '
                    f'sdata.start_30_days(track="{again}", start_again=True); to switch to the '
                    f'{_track_title(other)}, type sdata.start_30_days(track="{other}", start_again=True).'))
        return None
    progress = dict(track=track, started=_date_today(), opened={})
    info = TRACKS[track]
    print(_wrap(f"Welcome to 30 days of simplydata: the {info['title']}, {info['who']}."))
    print(_wrap(info["about"]))
    print()
    print(_wrap("Each day takes about an hour: a 5-minute warm-up of questions on earlier days (answer them "
                "from memory: trying to remember is what makes words stick), a lesson or two, and a task to do "
                "without looking back."))
    print()
    print("  sdata.today()       shows the day to do next; type it every day")
    print("  sdata.answers()     the answers to the warm-up and the task")
    print("  sdata.day()         all 30 days, with the ones you have opened ticked")
    print("  sdata.next_day()    goes on to the next day now, to do two in one go")
    print()
    print(_wrap("Missed a few days? That is fine: sdata.today() carries on where you stopped."))
    print(_wrap(f"Your progress is kept only on this computer, in {progress_file()}."))
    print(_wrap(f"The printable guide, with every day and a calendar to tick: {GUIDE_URL}"))
    print()
    _open_day(progress, 1, [])
    return None


def today(follow_calendar=False) -> None:
    """Show today's day of the 30 days of simplydata: warm-up questions, what to do and a task. sdata.today()

    The day after the one you last opened comes next, so a day you miss is never lost: today() carries
    on where you stopped (on the same date it shows the same day again). If you have fallen more than
    a day behind the calendar it says so; follow_calendar=True jumps to the calendar's day instead:
    sdata.today(follow_calendar=True). Not started yet? It shows how to start, and day 1.
    """
    progress = _read_progress()
    if progress is None:
        _print_not_started()
        _print_day(plan("python", 1), "To start, type sdata.start_30_days().")
        return None
    date = _date_today()
    calendar = max(1, (date - progress["started"]).days + 1)
    opened = progress["opened"]
    last = max(opened, default=0)
    notes = []
    if follow_calendar:
        number = max(min(calendar, DAYS_IN_A_TRACK), 1)
        if number > last + 1:
            skipped = f"Day {last + 1} was" if number == last + 2 else f"Days {last + 1} to {number - 1} were"
            notes.append(f"Following the calendar: day {number}. {skipped} not opened; sdata.day(n) shows any day.")
    elif last == 0:
        number = 1
    elif opened[last] == date:
        number = last
    else:
        number = last + 1
        if calendar - number >= 2 and number <= DAYS_IN_A_TRACK:
            notes.append(f"By the calendar it is day {min(calendar, DAYS_IN_A_TRACK)}. You last opened day {last}, "
                         f"on {_when(opened[last])}. The days build on each other, so today carries on with day "
                         f"{number}: nothing is lost by taking a break. To follow the calendar instead, type "
                         "sdata.today(follow_calendar=True).")
    if number > DAYS_IN_A_TRACK:
        _print_finished(progress)
        return None
    _open_day(progress, number, notes)
    return None


def next_day() -> None:
    """Go on to the next day now, without waiting for tomorrow. sdata.next_day()

    It opens the day after the last one you opened, so two days can be done in one go. Tomorrow,
    sdata.today() carries on from there.
    """
    progress = _read_progress()
    if progress is None:
        _print_not_started()
        _print_day(plan("python", 1), "To start, type sdata.start_30_days().")
        return None
    number = max(progress["opened"], default=0) + 1
    if number > DAYS_IN_A_TRACK:
        _print_finished(progress)
        return None
    _open_day(progress, number, [])
    return None


def _chosen_track(track) -> str:
    if track is not None:
        return _track(track)
    progress = _read_progress()
    return progress["track"] if progress else "python"


def day(number=None, track=None) -> None:
    """Show any day of the 30 days, without changing where you are. sdata.day(5)

    sdata.day() on its own lists all 30 days, with a tick for each day you have opened. The track is
    the one you started, else "python": sdata.day(1, track="beginner") shows the other one.
    """
    track = _chosen_track(track)
    if number is not None:
        number = _day_number(number, "day")
        _print_day(plan(track, number), _next_footer(track, number, browsing=True))
        return None
    progress = _read_progress()
    opened = progress["opened"] if progress and progress["track"] == track else {}
    heading = f"30 days of simplydata ({_track_title(track)}, {TRACKS[track]['who']})"
    print(heading)
    print("=" * min(len(heading), WIDTH))
    for n, (title, _, units, kind) in enumerate(DAYS[track], start=1):
        tick = "[x]" if n in opened else "[ ]"
        lessons = sorted({les for u in units for les in UNITS[u]["lessons"]})
        where = f"  (lesson {', '.join(map(str, lessons))})" if lessons and kind == "learn" else ""
        mark = {"review": "  - review", "project": "  - project", "final": "  - final test"}.get(kind, "")
        print(_wrap(f"{title}{where}{mark}", "            ", f"{tick} {n:>2}. "))
    print()
    if progress is None:
        print(_wrap('Not started yet: sdata.start_30_days() starts the Python track, '
                    'sdata.start_30_days(track="beginner") the beginner track.'))
    else:
        print(_wrap(f"Started on {_when(progress['started'])}. Type sdata.today() for the day to do next, "
                    "or sdata.day(n) to look at any day."))
    return None


def answers(number=None, track=None) -> None:
    """The answers to a day's warm-up questions, and a worked answer to its task. sdata.answers(5)

    sdata.answers() on its own gives the answers to the day you opened last. The track is the one you
    started, else "python": sdata.answers(5, track="beginner").
    """
    track = _chosen_track(track)
    if number is None:
        progress = _read_progress()
        number = max(progress["opened"], default=1) if progress else 1
    number = _day_number(number, "answers")
    _print_answers(plan(track, number))
    return None
