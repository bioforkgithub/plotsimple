"""Learning plotsimple: ps.words(), ps.help("bars") and ps.lesson(1) ... ps.lesson(8).

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

WIDTH = 78

# ---------------------------------------------------------------- the grouping
# "ps.name" is a function you call on ps; "name=" is an option used inside chart steps;
# a plain name is a step on a chart (chart.name(...)). Other spellings of a word
# (bar, scatter, color ...) are found by themselves and shown next to it.
GROUPS = [
    ("Start, show and save", ["ps.chart", "ps.example", "show", "save"]),
    ("Chart types", ["bars", "lines", "dots", "area", "histogram", "boxes", "violins",
                     "heatmap", "pie", "polar", "contour", "surface", "dots_3d"]),
    ("Statistics", ["show=", "error=", "error_bars", "fit_line", "fit_curve", "smooth_line",
                    "density", "count_bars", "all_points", "side_histograms", "ps.pairs"]),
    ("Lines and shading", ["line_at_x", "line_at_y", "shade_between_x", "shade_between_y"]),
    ("Words on the chart", ["title_top", "caption_bottom", "label_x", "label_y", "note",
                            "numbers_on_bars", "font_size"]),
    ("Axes", ["range_x", "range_y", "log_scale_x", "log_scale_y", "ticks_x", "ticks_y",
              "rotate_labels_x", "rotate_labels_y", "second_y", "flip_x", "flip_y"]),
    ("Colours and look", ["colour", "colours", "style", "size", "show_grid", "hide_grid",
                          "legend_top", "legend_bottom", "legend_left", "legend_right",
                          "hide_legend", "colour_bar_label"]),
    ("Several charts", ["one_per", "ps.side_by_side", "ps.one_above_other", "ps.grid",
                        "zoom_box", "animate"]),
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
            'show="total" adds them all up. show="count" counts the rows (bars only).'
        ),
        words=["bars", "lines"],
        example='chart.bars(x="class", y="marks", show="median")',
    ),
    "error=": dict(
        about="Error bars on top of averages: how spread out the values are.",
        text=(
            'error="sd" is the standard deviation: how far values usually are from the '
            'average. error="se" is the standard error: how sure we are about the average '
            'itself. error="ci" is the 95 % confidence interval: the range the true average '
            'is very likely to be in. Lines get a shaded band instead of bars.'
        ),
        words=["bars", "lines"],
        example='chart.bars(x="class", y="marks", error="sd")',
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
    "error=": _ex("marks", 'chart.bars(x="class", y="marks", error="sd")', "error"),
    "error_bars": _ex("marks", 'chart.lines(x="subject", y="marks", colour="class")\nchart.error_bars("se")', "error_bars"),
    "fit_line": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.fit_line(band=True, show_equation=True)', "fit_line"),
    "fit_curve": _ex("weather", 'chart.dots(x="month_number", y="temperature", colour="city")\nchart.fit_curve(degree=2)', "fit_curve"),
    "smooth_line": _ex("weather", 'chart.dots(x="month_number", y="rain", colour="city")\nchart.smooth_line()', "smooth_line"),
    "density": _ex("marks", 'chart.density(x="marks", colour="class")', "density"),
    "count_bars": _ex("fruit", 'chart.count_bars(x="fruit", colour="shop")', "count_bars"),
    "all_points": _ex("marks", 'chart.boxes(x="class", y="marks")\nchart.all_points(x="class", y="marks")', "all_points"),
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
    "second_y": _ex("weather", 'chart.lines(x="month", y="temperature")\nchart.second_y("rain", chart="bars", label="Rain (mm)")', "second_y"),
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
    "colour_bar_label": _ex("island", 'chart.contour(x="east", y="north", z="height")\nchart.colour_bar_label("Height (m)")', "colour_bar_label"),
    # several charts
    "one_per": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.one_per("class")', "one_per"),
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
        page = ps.grid(a, b, c, d, columns=2)
        page.title_top("Four ways to look at the marks")
        page.save("grid.png")"""),
    "zoom_box": _ex("marks", 'chart.dots(x="hours", y="marks")\nchart.zoom_box(x_from=2, x_to=4, y_from=40, y_to=60)', "zoom_box"),
    "animate": textwrap.dedent("""\
        import plotsimple as ps
        growth = ps.example("growth")
        chart = ps.chart(growth)
        chart.bars(x="country", y="population")
        chart.range_y(0, 50)
        chart.animate(over="year")
        chart.save("growth.gif")"""),
    # help
    "ps.words": "import plotsimple as ps\nps.words()",
    "ps.help": 'import plotsimple as ps\nps.help("bars")\nps.help("show=")',
    "ps.lesson": "import plotsimple as ps\nps.lesson()\nps.lesson(1)",
}


# ---------------------------------------------------------------- finding words
def _ps():
    import plotsimple  # late import: plotsimple imports this file

    return plotsimple


def _chart_class():
    from .chart import Chart

    return Chart


def _function(key: str):
    """The real function behind a word in GROUPS (None for options such as "show=")."""
    if key.endswith("="):
        return None
    if key.startswith("ps."):
        return getattr(_ps(), key[3:])
    return getattr(_chart_class(), key)


def _all_keys() -> list[str]:
    return [k for _, keys in GROUPS for k in keys]


def _group_of(key: str) -> str:
    return next(g for g, keys in GROUPS if key in keys)


def _aliases() -> dict[str, list[str]]:
    """Other names for the same step: {"bars": ["bar"], "dots": ["scatter"], ...}"""
    chart_cls = _chart_class()
    found: dict[str, list[str]] = {}
    keys = [k for k in _all_keys() if not k.endswith("=") and not k.startswith("ps.")]
    for name in chart_cls._word_names():
        if name in keys:
            continue
        func = getattr(chart_cls, name)
        for key in keys:
            if getattr(chart_cls, key) is func:
                found.setdefault(key, []).append(name)
    return found


def _call_form(key: str) -> str:
    if key.endswith("="):
        return key
    if key.startswith("ps."):
        return f"{key}()"
    return f"chart.{key}()"


def _find_call(text: str, prefix_word: str) -> str | None:
    """Find the first call like chart.bars(...) in text, brackets and quotes included."""
    m = re.search(rf"(?:chart|ps|page)\.{re.escape(prefix_word)}\(", text)
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
    word = key[3:] if key.startswith("ps.") else key
    doc = inspect.getdoc(_function(key)) or ""
    call = _find_call(doc, word) or _find_call(HELP_EXAMPLES.get(key, ""), word)
    return call or _call_form(key)


_TABLE = re.compile(r'ps\.example\("(\w+)"\)')


def _example_table(key: str) -> str | None:
    """The example table a word's examples use: named in the docstring, else in the help example.

    A docstring names its table like this: chart.lines(...) (with ps.example("weather")).
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
    word = key[3:] if key.startswith("ps.") else key
    call = _find_call(first, word)
    if call:
        first = first.split(call)[0]
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
    pad = max(len(k) for k in _all_keys()) + 2
    print("plotsimple words. Type ps.help(\"bars\") to learn more about any of them.")
    print('Most words are steps on a chart: chart = ps.chart(table), then chart.bars(...).')
    print('The examples use the practice tables; ps.help("word") shows which one, ready to run.')
    for group, keys in GROUPS:
        print()
        print(group)
        print("-" * len(group))
        for key in keys:
            line = f"  {key:<{pad}}{_short_example(key)}"
            if key in aliases:
                line += "   (also: " + ", ".join(aliases[key]) + ")"
            print(line)
    print()
    print(f"Missing a word? Ask for it at {ISSUES_URL}")


# ---------------------------------------------------------------- ps.help()
def _normalise(word: str) -> str:
    w = str(word).strip()
    for start in ("ps.", "chart.", "page.", "."):
        if w.startswith(start):
            w = w[len(start):]
    if w.endswith("()"):
        w = w[:-2]
    return w.strip()


def _lookup(word) -> str | None:
    """Turn what the user typed into a key of GROUPS, or None."""
    w = _normalise(word)
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
        if w in names:
            return key
    if w in ("chart", "Chart"):
        return "ps.chart"
    return None


def _options_text(key: str) -> list[str]:
    func = _function(key)
    if func is None:
        return []
    params = [p for p in inspect.signature(func).parameters.values() if p.name != "self"]
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
        elif p.default is None:
            lines.append(f"  {p.name}=")
        else:
            normal = f'"{p.default}"' if isinstance(p.default, str) else repr(p.default)
            lines.append(f"  {p.name + '=':<{width}}normally {normal}")
    return lines


def _print_help_overview() -> None:
    print(textwrap.dedent("""\
        How to get help in plotsimple

          ps.help("bars")     explains one word: what it does, its options and an example
          ps.help("show=")    explains an option you use inside a step
          ps.words()          lists every word, with a short example of each
          ps.lesson()         lists the lessons: learn the whole of plotsimple in a day
          ps.example()        lists the made-up tables you can practise with

        You can type the word with or without "chart." in front and with or without
        brackets: ps.help("bars"), ps.help("chart.bars") and ps.help("bars()") all work.
        """))


def help(word=None) -> None:  # noqa: A001 - meant to be ps.help
    """Explain one word: what it does, its options and an example you can run. ps.help("bars")

    ps.help() on its own explains how to get help.
    """
    if word is None:
        _print_help_overview()
        return None
    if not isinstance(word, str):
        name = getattr(word, "__name__", None)
        if name is None or _lookup(name) is None:
            return builtins.help(word)  # not a plotsimple word: Python's own help
        word = name
    key = _lookup(word)
    if key is None:
        known = sorted({_normalise(k) for k in _all_keys()} | {a for v in _aliases().values() for a in v})
        lines = [f'plotsimple has no word called "{word}".']
        guesses = suggest(_normalise(word), known)
        if guesses:
            lines.append("Did you mean " + " or ".join(f'ps.help("{g}")' for g in guesses) + "?")
        lines.append("Type ps.words() to see every word.")
        lines.append(f"Missing something? Ask for it at {ISSUES_URL}")
        print("\n".join(lines))
        return None

    title = _call_form(key)
    print(title)
    print("=" * len(title))
    print(f"Group: {_group_of(key)}")
    aliases = _aliases().get(key)
    if aliases:
        print("Same word, other spelling: " + ", ".join(f"chart.{a}()" for a in aliases))
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
            "The legend can go on any side: legend_top, legend_bottom, legend_left, legend_right, "
            "or hide_legend to take it away.",
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
                ('chart.all_points(x="class", y="marks")', "Also show every single mark as a dot, "
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
        words=["lines", "dots", "histogram", "boxes", "all_points", "pie", "violins", "area"],
    ),
    dict(
        title="Statistics: averages, error bars and fit lines",
        learn=[
            "what a bar shows when a group has many values",
            "how to add error bars",
            "how to draw the straight line that fits some dots best",
        ],
        parts=[
            ("An average is all the values added up, divided by how many there are. Error bars "
             "show how far the values spread around it.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.bars(x="class", y="marks", show="average", error="sd")',
                 'One bar per class, as tall as the average mark. error="sd" adds error bars: the '
                 "standard deviation, which says how far marks usually are from the average."),
                ("chart.numbers_on_bars()", "Write the average on each bar."),
                ('chart.save("lesson5_average.png")', "Save it."),
            ]),
            ("A fit line is the straight line that goes as close as possible to all the dots.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "One dot per test: hours of study and the mark."),
                ("chart.fit_line(band=True, show_equation=True)", "Add the best straight line. "
                 "band=True shades where the line could be (95 % confidence). show_equation=True "
                 "writes its equation and R² (1 means the dots sit right on the line, 0 means "
                 "no pattern at all)."),
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
            'show= can be "average", "median" (the middle value), "total" (all added up) or '
            '"count" (how many rows).',
            'error= can be "sd" (standard deviation), "se" (standard error: how sure we are of the '
            'average) or "ci" (95 % confidence interval).',
            "fit_curve(degree=2) fits a curve with one bend; smooth_line() follows the dots freely.",
        ],
        task=[
            'Change show="average" to show="median". Do the bars change much?',
            'Change error="sd" to error="se". Are the error bars bigger or smaller?',
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
            ("one_per draws the same chart once for each group.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.fit_line()", "Add a fit line."),
                ('chart.one_per("class")', "Make one small chart for each class, with the same axes "
                                           "so they are easy to compare."),
                ('chart.save("lesson7_one_per.png")', "Save it."),
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
                ("page = ps.grid(a, b, c, d, columns=2)", "Put all four in a grid with 2 columns."),
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
            'chart.one_per(row="class", column="subject") makes a table of small charts.',
        ],
        task=[
            'Change chart.one_per("class") to chart.one_per("subject").',
            "Use ps.side_by_side(a, b) instead of ps.one_above_other(a, b).",
            'Make ps.pairs of the weather table with colour="city".',
        ],
        words=["one_per", "ps.one_above_other", "ps.grid", "ps.pairs", "ps.side_by_side", "fit_line"],
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
                ('chart.second_y("rain", chart="bars", label="Rain (mm)")', "Add rain as bars, with "
                                                                            "its own y-axis on the "
                                                                            "right."),
                ('chart.label_y("Temperature (°C)")', "Name the left y-axis."),
                ('chart.save("lesson8_second_y.png")', "Save it."),
            ]),
            ("A zoom box shows a small part of the chart close up.", [
                ("import plotsimple as ps", "Load plotsimple."),
                ('marks = ps.example("marks")', "Get the marks table."),
                ("chart = ps.chart(marks)", "Start a chart."),
                ('chart.dots(x="hours", y="marks")', "Dots of hours and marks."),
                ("chart.zoom_box(x_from=2, x_to=4, y_from=40, y_to=60)", "Show the part from 2 to 4 "
                                                                         "hours and 40 to 60 marks, "
                                                                         "bigger, in a box."),
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
                ('chart.colour_bar_label("Rain (mm)")', "Name the colour bar."),
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
                ('chart.colour_bar_label("Height (m)")', "... and the name of the colour bar."),
                ('chart.note("X marks the spot", x=10, y=4, arrow=True)', "A note with an arrow."),
                ('chart.style("presentation")', "6. Look: big and clear for showing to others."),
                ('chart.save("lesson8_my_program.png")', "7. Save."),
            ]),
        ],
        notes=[
            "That is the whole of plotsimple! ps.words() lists every word and ps.help(\"word\") "
            "explains any of them.",
            'Your own data: save it from Excel as .xlsx or .csv, then chart = ps.chart("my_file.xlsx"). '
            "If a column name is spelt wrong, plotsimple tells you the right one.",
        ],
        task=[
            "Write your own program, from an empty file, for a question you care about.",
            "Use any example table, or your own. Give it a title, axis names and a note.",
            "Put two of your charts side by side and save the page as a PDF.",
        ],
        words=["second_y", "zoom_box", "heatmap", "colour_bar_label", "size", "animate",
               "contour", "note", "style", "ps.words", "ps.help"],
    ),
]


def _wrap(text: str, indent: str = "", width: int = WIDTH) -> str:
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def lesson_code(number: int) -> list[str]:
    """The code of each part of a lesson, ready to run (used by the tests)."""
    return ["\n".join(code for code, _ in steps) for _, steps in LESSONS[number - 1]["parts"]]


def _list_lessons() -> None:
    print("Learn plotsimple in one day: 8 lessons of about 30 to 45 minutes.\n")
    for i, les in enumerate(LESSONS, start=1):
        print(f"  ps.lesson({i})   {les['title']}")
    print("\nStart with ps.lesson(1). Each lesson ends with a small task to try yourself.")


def lesson(number=None) -> None:
    """A lesson: an example to run, what each line does, and a task. ps.lesson(1)

    ps.lesson() on its own lists all 8 lessons.
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
