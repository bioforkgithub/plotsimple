"""The Story: one object for a whole analysis, from opening a file to a saved figure.

    story = ps.start(file="results.xlsx")
    story.remove_rows_with_empty_cells()
    story.keep_rows("score", above=50)
    story.draw_bars(x="class", y="score")
    story.title_top("Good scores")
    story.save_chart("results.png")
    story.save_table("clean.xlsx")

A story is a thin layer over a Table (every table word) and a list of Charts (every chart
word). It adds only what one object for both needs:

* chart types get draw_ in front (draw_bars, draw_histogram ...), so every line of a story
  starts with a verb and you can see where the chart part begins;
* chart decorations (title_top, label_y, fit_line ...) change the current chart: the one
  you drew last;
* save_table and save_chart (and plain save, which picks by the file type);
* one history over table and chart steps: undo(), steps() and start_again();
* new_chart() and put_charts_together() for several charts.

The snapshot rule: a chart uses the table as it was when its first draw_ step was written.
Later table steps do not change charts that are already drawn, so the story reads top to
bottom like a story. For the table as it is now, start a new chart with new_chart().
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from ._core import SAVE_FORMATS
from ._errors import MissingWordError, PlotsimpleError, choice_error, friendly, missing_word_error, suggest
from .chart import Chart
from .table import Table
from .table._base import HISTORY_LIMIT, SAVE_TYPES

# Chart types: on a story they are drawn with draw_ in front (draw_bars, draw_histogram ...).
DRAW_WORDS = ("bars", "lines", "dots", "area", "histogram", "boxes", "violins", "heatmap", "pie",
              "polar", "contour", "surface", "dots_3d", "density", "count_bars", "all_dots")
# Other spellings of chart types that a chart understands (chart.bar, chart.scatter ...).
_CHART_TYPE_SPELLINGS = {"bar": "bars", "line": "lines", "scatter": "dots", "box": "boxes", "violin": "violins"}
# Table words a story has in its own way (one history over table and chart steps, two saves).
_OWN_TABLE_WORDS = {"save", "undo", "steps", "start_again", "chart"}
# Chart words a story has in its own way.
_OWN_CHART_WORDS = {"save", "show"} | set(DRAW_WORDS) | set(_CHART_TYPE_SPELLINGS)
# Words that also work after put_charts_together(): they change the whole page.
_PAGE_WORDS = ("title_top", "style", "size")

# How a chart type is said in steps(): "draw bars of marks for each class".
_DRAW_TEXT = {
    "bars": "bars", "lines": "lines", "dots": "dots", "area": "an area chart", "histogram": "a histogram",
    "boxes": "boxes", "violins": "violins", "heatmap": "a heatmap", "pie": "a pie chart",
    "polar": "a polar chart", "contour": "a contour map", "surface": "a 3D surface",
    "dots_3d": "dots in 3D", "density": "a density curve", "count_bars": "bars that count rows",
    "all_dots": "all the dots",
}
_FOR_EACH = {"bars", "boxes", "violins", "all_dots"}
_AGAINST = {"lines", "dots", "area"}

# A few chart words read better in steps() with their own sentence.
_DECORATION_TEXT = {
    "title_top": 'title on top: {text}',
    "caption_bottom": 'caption at the bottom: {text}',
    "label_x": "label the x axis {text}",
    "label_y": "label the y axis {text}",
    "label_colour_bar": "label the colour bar {text}",
    "line_at_x": "line at x = {value}",
    "line_at_y": "line at y = {value}",
    "shade_between_x": "shade x between {low} and {high}",
    "shade_between_y": "shade y between {low} and {high}",
    "range_x": "x axis from {low} to {high}",
    "range_y": "y axis from {low} to {high}",
    "log_scale_x": "log scale on the x axis",
    "log_scale_y": "log scale on the y axis",
    "legend_top": "legend at the top",
    "legend_bottom": "legend at the bottom",
    "legend_left": "legend on the left",
    "legend_right": "legend on the right",
    "fit_line": "fit a straight line",
    "numbers_on_bars": "numbers on the bars",
}

_CALL = re.compile(r"(?<![\w.])(chart|table)\.(\w+)\(")


def _to_story(text: str) -> str:
    """chart.bars(...) -> story.draw_bars(...), table.keep_rows(...) -> story.keep_rows(...)."""

    def fix(m):
        thing, word = m.groups()
        word = _CHART_TYPE_SPELLINGS.get(word, word) if thing == "chart" else word
        if thing == "chart" and word in DRAW_WORDS:
            word = "draw_" + word
        elif word == "save":
            word = "save_chart" if thing == "chart" else "save_table"
        elif thing == "chart" and word == "show":
            word = "show_chart"
        return f"story.{word}("

    return _CALL.sub(fix, text)


def _said(value) -> str:
    """A value as steps() says it: text in quotes, numbers and lists as they are."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return ", ".join(_said(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return repr(value)


def _given(func, args, kwargs) -> dict:
    """The options the user gave, by name, in the order of the word's brackets."""
    try:
        bound = inspect.signature(func).bind_partial(None, *args, **kwargs)
    except TypeError:
        return dict(kwargs)
    given = dict(bound.arguments)
    given.pop(next(iter(given)), None)  # self
    return given


def _draw_sentence(kind: str, given: dict, columns) -> str:
    def col(name):
        value = given.get(name)
        return value if isinstance(value, str) else None

    text = "draw " + _DRAW_TEXT[kind]
    x, y = col("x"), col("y")
    show = col("show")
    measure = f"the {show} " if show and show != "average" else ""
    if kind == "heatmap":
        if col("value"):
            text += f" of {col('value')}"
        if x and y:
            text += f" by {x} and {y}"
    elif kind == "pie":
        if col("labels") and col("values"):
            text += f" of {col('values')} for each {col('labels')}"
        elif col("labels"):
            text += f" counting each {col('labels')}"
    elif kind == "polar":
        if col("distance") and col("angle"):
            text += f" of {col('distance')} against {col('angle')}"
    elif kind in ("contour", "surface", "dots_3d"):
        if col("z") and x and y:
            text += f" of {col('z')} over {x} and {y}"
    elif x and y:
        text += f" of {measure}{y} " + ("for each" if kind in _FOR_EACH else "against") + f" {x}"
    elif x:
        text += f" of {x}"
    elif y:
        text += f" of {measure}{y}"
    colour = col("colour")
    if colour:
        text += f", coloured by {colour}" if colour in columns else f", in {colour}"
    if col("error"):
        text += f", with {col('error')} bars"
    return text


def _decoration_sentence(name: str, given: dict) -> str:
    template = _DECORATION_TEXT.get(name)
    if template:
        try:
            return template.format(**{k: _said(v) for k, v in given.items()})
        except (KeyError, ValueError):
            pass
    words = name.replace("_", " ")
    parts = []
    for i, (key, value) in enumerate(given.items()):
        parts.append(_said(value) if i == 0 else f"{key.replace('_', ' ')} {_said(value)}")
    return words + (" " + ", ".join(parts) if parts else "")


def _with_story_words(exc: PlotsimpleError) -> PlotsimpleError:
    """The same error, with its examples written as story words."""
    message = str(exc)
    new = _to_story(message)
    if new == message:
        return exc
    try:
        return type(exc)(new)
    except TypeError:  # pragma: no cover - an error class with other arguments
        return exc


def _unwrap(value):
    """A story given to a table word (join_with, add_rows_from) stands for its table."""
    return value._table if isinstance(value, Story) else value


class Story:
    """A story: one table and its charts, told step by step. Start one with ps.start(...)."""

    _is_plotsimple_story = True
    _OLD_WORDS: dict = {}  # a story is new: no old names yet

    def __init__(self, _table=None):
        if not getattr(_table, "_is_plotsimple_table", False):
            raise PlotsimpleError('Start a story with ps.start(...), for example ps.start(example="marks") '
                                  'or ps.start(file="results.xlsx").')
        object.__setattr__(self, "_table", _table)
        object.__setattr__(self, "_charts", [])    # every chart drawn so far, the current one last
        object.__setattr__(self, "_mode", "empty")  # "empty": the next draw_ starts a chart;
        #                                             "chart": the last chart is current; "page": the page is
        object.__setattr__(self, "_page", None)
        object.__setattr__(self, "_history", [])   # [what was done, chart number or None, how to undo it]

    # ------------------------------------------------------------------ history
    def _snapshot(self) -> dict:
        """Everything a step can change, so undo() can put it back. Cheap: tables are not
        changed in place, so only references and short lists are kept."""
        return dict(
            frame=self._table._frame,
            table_history=list(self._table._history),
            charts=[(c, list(c._steps), dict(c._settings)) for c in self._charts],
            mode=self._mode,
            page=self._page,
            page_settings=dict(self._page._settings) if self._page is not None else None,
        )

    def _restore(self, snap: dict) -> None:
        object.__setattr__(self._table, "_frame", snap["frame"])
        object.__setattr__(self._table, "_history", snap["table_history"])
        for chart, steps, settings in snap["charts"]:
            chart._steps, chart._settings, chart._cache = steps, settings, None
        object.__setattr__(self, "_charts", [c for c, _, _ in snap["charts"]])
        object.__setattr__(self, "_mode", snap["mode"])
        object.__setattr__(self, "_page", snap["page"])
        if snap["page"] is not None:
            snap["page"]._settings = snap["page_settings"]

    def _record(self, what: str, chart_number, snap: dict) -> None:
        self._history.append([what, chart_number, snap])
        if len(self._history) > HISTORY_LIMIT:  # keep the list of steps, forget how to undo old ones
            self._history[-HISTORY_LIMIT - 1][2] = None

    # ------------------------------------------------------------------ the current chart
    def _has_chart(self) -> bool:
        return self._mode != "empty"

    def _current(self, word: str):
        """The chart (or page) a chart word changes. word is said in the error."""
        if self._mode == "chart":
            return self._charts[-1]
        if self._mode == "page":
            return self._page
        if self._charts:
            raise PlotsimpleError(
                f"Draw a chart first, for example story.draw_bars(x=..., y=...).\n"
                f"story.new_chart() started a new, empty chart, and story.{word}() works on the chart "
                f"you drew last.\nstory.undo() takes you back to chart {len(self._charts)}.")
        raise PlotsimpleError(
            f"Draw a chart first, for example story.draw_bars(x=..., y=...).\n"
            f"story.{word}() works on the chart you drew last.")

    def _chart_for_page(self, step: str, number: int):
        """The chart this story gives to ps.grid / ps.side_by_side / ps.one_above_other."""
        if self._mode == "page":
            raise PlotsimpleError(
                f"Story {number} given to ps.{step}() has its charts put together on a page already.\n"
                "Give stories whose last chart is a single chart, or save the page with story.save_chart(...).")
        try:
            return self._current(step)
        except PlotsimpleError:
            raise PlotsimpleError(
                f"Story {number} given to ps.{step}() has no chart yet. "
                "Draw one first, for example story.draw_bars(x=..., y=...).") from None

    def _explain_old_table(self, exc: PlotsimpleError, chart) -> PlotsimpleError:
        """A column the chart cannot find, but the table has now: explain the snapshot rule."""
        found = re.match(r'Column "(.+?)" not found', str(exc))
        if found and found.group(1) in self._table._columns and found.group(1) not in chart._data.columns:
            return type(exc)(
                f"{exc}\nThis chart uses the table as it was when you drew it first, and that table had "
                f'no column "{found.group(1)}" yet.\nTo draw the table as it is now, start a new chart '
                "first: story.new_chart()")
        return exc

    # ------------------------------------------------------------------ doing a step
    def _table_step(self, name, args, kwargs):
        args = [_unwrap(a) for a in args]
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        snap = self._snapshot()
        history = self._table._history
        last = history[-1] if history else None
        try:
            result = getattr(self._table, name)(*args, **kwargs)
        except PlotsimpleError as exc:
            raise _with_story_words(exc) from None
        history = self._table._history
        if history and history[-1] is not last:  # the table changed: one step in the story
            self._record(history[-1][0], None, snap)
        if result is self._table:
            return self
        if getattr(result, "_is_plotsimple_table", False):
            return Story(result)  # a summary or test: a new story, so it can be drawn
        return result

    def _draw_step(self, kind, args, kwargs):
        if self._mode == "page":
            raise PlotsimpleError(
                "The charts are put together on one page now.\n"
                f"Start another chart first with story.new_chart(), then story.draw_{kind}(...).\n"
                "Or story.undo() to take the page apart again.")
        snap = self._snapshot()
        new = self._mode == "empty"
        chart = Chart(self._table) if new else self._charts[-1]
        method = getattr(Chart, kind)
        try:
            method(chart, *args, **kwargs)
        except PlotsimpleError as exc:
            exc = exc if new else self._explain_old_table(exc, chart)
            raise _with_story_words(exc) from None
        if new:
            self._charts.append(chart)
            object.__setattr__(self, "_mode", "chart")
        self._record(_draw_sentence(kind, _given(method, args, kwargs), chart._data.columns),
                     len(self._charts), snap)
        return self

    def _chart_step(self, name, args, kwargs):
        target = self._current(name)
        if self._mode == "page" and name not in _PAGE_WORDS:
            raise PlotsimpleError(
                f"The charts are put together on one page now, so story.{name}() has no chart to change.\n"
                "On the page, only " + ", ".join(f"story.{w}()" for w in _PAGE_WORDS) + " work.\n"
                f"Use story.{name}() before put_charts_together(), or story.undo() to take the page apart.")
        snap = self._snapshot()
        method = getattr(type(target), name)
        try:
            result = method(target, *args, **kwargs)
        except PlotsimpleError as exc:
            exc = self._explain_old_table(exc, target) if self._mode == "chart" else exc
            raise _with_story_words(exc) from None
        what = _decoration_sentence(name, _given(method, args, kwargs))
        if self._mode == "page":
            what += " (the whole page)"
        self._record(what, len(self._charts) if self._mode == "chart" else None, snap)
        return self if result is target else result

    # ------------------------------------------------------------------ the story's own words
    @friendly
    def new_chart(self):
        """Start another chart; the next draw_ step draws it. story.new_chart() (with ps.example("marks"))

        The new chart uses the table as it is when you draw it, so table steps written after
        your first chart show up in the new one. Chart words such as title_top then change
        the new chart. story.put_charts_together() puts all the charts on one page.
        """
        if self._mode == "empty":
            print("A new chart starts with the next draw_ step, for example story.draw_bars(x=..., y=...).")
            return self
        snap = self._snapshot()
        object.__setattr__(self, "_mode", "empty")
        self._record(f"start a new chart (chart {len(self._charts) + 1})", None, snap)
        return self

    @friendly
    def put_charts_together(self, per_row=2, letters=False, same_x=False, same_y=False):
        """All the charts of the story on one page. story.put_charts_together(per_row=2, letters=True)
        (with ps.example("marks"))

        per_row=2 puts 2 charts in each row. letters=True writes A, B, C ... at the top left
        of each chart, as figures in papers do. same_x=True / same_y=True give the charts the
        same axis range. Then story.save_chart("figure.png") saves the whole page, and
        title_top, style and size change the whole page.
        """
        if len(self._charts) < 2:
            raise PlotsimpleError(
                ".put_charts_together() needs at least two charts.\n"
                "Draw one, then story.new_chart() and draw another, for example:\n"
                '    story.draw_bars(x="class", y="marks")\n    story.new_chart()\n'
                '    story.draw_dots(x="hours", y="marks")')
        from .words.layout import grid

        page = grid(*self._charts, per_row=per_row, same_x=same_x, same_y=same_y, letters=letters)
        snap = self._snapshot()
        object.__setattr__(self, "_page", page)
        object.__setattr__(self, "_mode", "page")
        n = len(self._charts)
        what = f"put charts 1 to {n} together, {min(per_row, n)} per row"
        self._record(what + (", with letters" if letters else ""), None, snap)
        return self

    @friendly
    def save_chart(self, file_name, quality="normal"):
        """Save the chart you drew last (or the page of charts). story.save_chart("results.png")
        (with ps.example("marks"))

        The end of the file name decides the type: .png (a picture), .pdf and .svg (sharp at
        any size), .jpg, .tif, and .gif for animations. quality="high" saves a sharper picture.
        After story.put_charts_together(...) it saves the whole page.
        """
        if not self._charts:
            raise PlotsimpleError("There is no chart to save yet. Draw one first, for example "
                                  "story.draw_bars(x=..., y=...).")
        try:
            self._current("save_chart").save(file_name, quality=quality)
        except PlotsimpleError as exc:
            raise _with_story_words(exc) from None
        return self

    @friendly
    def save_table(self, file_name):
        """Save the table as it is now. story.save_table("clean.xlsx") (with ps.example("marks"))

        The end of the file name decides the type: .xlsx (Excel), .csv, .tsv, .parquet and .json.
        """
        try:
            self._table.save(file_name)
        except PlotsimpleError as exc:
            raise _with_story_words(exc) from None
        return self

    @friendly
    def save(self, file_name, quality="normal"):
        """Save the table or the chart; the file type says which. story.save("results.png")
        (with ps.example("marks"))

        Table types (.xlsx, .csv, .tsv, .parquet, .json) save the table, picture types (.png,
        .pdf, .svg, .jpg, .tif, .gif) save the chart. story.save_table(...) and
        story.save_chart(...) say it in words, which is easier to read later.
        """
        path = Path(file_name)
        kind = path.suffix.lower().lstrip(".")
        if not kind:
            raise PlotsimpleError(
                f'Add a file type to the name, for example story.save_table("{path.name}.xlsx") '
                f'or story.save_chart("{path.name}.png").')
        if kind in SAVE_TYPES:
            if quality != "normal":
                raise PlotsimpleError(f'quality= is for pictures; a .{kind} file saves the table.')
            return self.save_table(file_name)
        if kind in SAVE_FORMATS:
            return self.save_chart(file_name, quality=quality)
        raise choice_error("save", "file type", kind, list(SAVE_TYPES) + list(SAVE_FORMATS))

    @friendly
    def show_chart(self):
        """Show the chart you drew last (or the page) in a window. story.show_chart()
        (with ps.example("marks"))

        In Jupyter, VS Code and Spyder the chart appears by itself.
        """
        self._current("show_chart").show()
        return self

    @friendly
    def undo(self):
        """Take the last step back: a table step or a chart step. story.undo() (with ps.example("marks"))

        Undo after draw_ takes that chart (or that layer of it) away again; undo after a
        table step gives back the table as it was.
        """
        if not self._history:
            print("Nothing to undo: this story has no steps yet.")
            return self
        what, _, snap = self._history[-1]
        if snap is None:
            print(f"plotsimple remembers how to undo the last {HISTORY_LIMIT} steps only, so "
                  f'"{what}" cannot be taken back. story.start_again() goes back to the start.')
            return self
        self._history.pop()
        self._restore(snap)
        print(f"Undone: {what}")
        return self

    @friendly
    def steps(self):
        """List every step of the story, table and chart steps, in order. story.steps()
        (with ps.example("marks"))
        """
        if not self._history:
            print("No steps yet.")
            return self
        several = any((n or 0) > 1 for _, n, _ in self._history)
        for i, (what, number, _) in enumerate(self._history, 1):
            print(f"{i}. {what}" + (f"  (chart {number})" if several and number else ""))
        return self

    @friendly
    def start_again(self):
        """Go back to the very start: the table as it was opened, and no charts.
        story.start_again() (with ps.example("marks"))

        It is a step too, so story.undo() takes it back.
        """
        snap = self._snapshot()
        object.__setattr__(self._table, "_frame", self._table._original)
        object.__setattr__(self._table, "_history", [])
        object.__setattr__(self, "_charts", [])
        object.__setattr__(self, "_mode", "empty")
        object.__setattr__(self, "_page", None)
        self._record("start again", None, snap)
        return self

    # ------------------------------------------------------------------ Python details
    def __len__(self):
        return len(self._table)

    def __repr__(self):
        rows, cols = self._table._frame.shape
        text = f"plotsimple story: a table of {rows} rows and {cols} columns"
        if self._charts:
            n = len(self._charts)
            text += f", {n} chart" + ("s" if n > 1 else "")
            if self._mode == "page":
                text += " (put together on one page)"
        return f"{text}\n{self._table._frame.head(10).to_native()!r}"

    def _repr_png_(self):
        """In Jupyter, a story with a chart shows its chart; otherwise its table."""
        return self._current("show")._repr_png_() if self._has_chart() else None

    def _repr_html_(self):
        return None if self._has_chart() else self._table._repr_html_()

    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        kind = _CHART_TYPE_SPELLINGS.get(name, name)
        if kind in DRAW_WORDS:
            raise MissingWordError(
                f"On a story, chart types start with draw_: story.draw_{kind}(...)\n"
                "So every line starts with a verb, and you can see where the chart part begins.")
        if name == "show":
            raise MissingWordError("A story can show two things: story.show_chart() shows the chart, and "
                                   "story.show_first_rows() shows the first rows of the table.")
        if name == "chart":
            raise MissingWordError("A story draws its charts itself: story.draw_bars(x=..., y=...).\n"
                                   "story.new_chart() starts another chart.")
        raise missing_word_error(name, self._word_names())

    def __setattr__(self, name, value):
        guesses = suggest(name, self._word_names())
        hint = f"story.{guesses[0]}(...)" if guesses else 'a step such as story.title_top("...")'
        raise PlotsimpleError(f"Steps are written with brackets, not with =.\n"
                              f"Instead of story.{name} = ..., write {hint}")


# ---------------------------------------------------------------- the words a story borrows
def _borrowed(name, target, doc, do):
    """A story word that does `do(self, args, kwargs)`, with the options and help of `target`."""

    def word(self, *args, **kwargs):
        return do(self, args, kwargs)

    word.__name__ = name
    word.__qualname__ = f"Story.{name}"
    word.__doc__ = doc
    word.__signature__ = inspect.signature(target)
    return friendly(word, renamed=getattr(target, "_renamed_options", None))


def _draw_doc(kind: str) -> str:
    doc = inspect.getdoc(getattr(Chart, kind)) or ""
    doc = "Draw " + doc[0].lower() + doc[1:]
    return (_to_story(doc) + f"\n\nThis is chart.{kind}() with draw_ in front, so the line starts with a verb.\n"
            f'Every option is the same: ps.help("{kind}") explains them all.')


def _add_words():
    for kind in DRAW_WORDS:
        setattr(Story, f"draw_{kind}", _borrowed(
            f"draw_{kind}", getattr(Chart, kind), _draw_doc(kind),
            lambda self, args, kwargs, kind=kind: self._draw_step(kind, args, kwargs)))
    for name in Table._word_names():
        if name in _OWN_TABLE_WORDS:
            continue
        target = getattr(Table, name)
        setattr(Story, name, _borrowed(
            name, target, _to_story(inspect.getdoc(target) or ""),
            lambda self, args, kwargs, name=name: self._table_step(name, args, kwargs)))
    made = {}
    for name in Chart._word_names():
        if name in _OWN_CHART_WORDS:
            continue
        target = getattr(Chart, name)
        if target in made:  # another spelling (color, colors ...): the same word
            setattr(Story, name, made[target])
            continue
        doc = _to_story(inspect.getdoc(target) or "") + "\n\nOn a story it changes the chart you drew last."
        made[target] = _borrowed(
            name, target, doc, lambda self, args, kwargs, name=target.__name__: self._chart_step(name, args, kwargs))
        setattr(Story, name, made[target])


_add_words()


# ---------------------------------------------------------------- ps.start
@friendly
def start(file=None, example=None, table=None, sheet=None, fast=False) -> Story:
    """Start a story: open a table, then tell what to do with it. ps.start(example="marks")

    ps.start(file="results.xlsx")              your own file (Excel, CSV, TSV, Parquet, JSON)
    ps.start(file="results.xlsx", sheet="2026") one sheet of an Excel file
    ps.start(example="marks")                  a made-up table to practise with (ps.example() lists them)
    ps.start(table=df)                         a pandas, Polars or plotsimple table you already have
    fast=True                                  use the fast engine (Polars), for big data

    Then every table word (story.keep_rows ...) and every chart word works on the story.
    Chart types start with draw_ (story.draw_bars ...), and words such as title_top change
    the chart you drew last.
    """
    if not isinstance(fast, bool):
        raise PlotsimpleError("fast= should be True or False.")
    given = [name for name, value in (("file", file), ("example", example), ("table", table)) if value is not None]
    if not given:
        raise PlotsimpleError(
            "ps.start() needs a table to start from, for example:\n"
            '    ps.start(file="results.xlsx")   your own file\n'
            '    ps.start(example="marks")       a made-up table to practise with\n'
            "    ps.start(table=df)              a pandas or Polars table you already have")
    if len(given) > 1:
        raise PlotsimpleError(f"ps.start() starts from one table, so give only one of "
                              f"{' and '.join(n + '=' for n in given)}.")
    if sheet is not None and file is None:
        raise PlotsimpleError('sheet= is for Excel files, for example ps.start(file="results.xlsx", sheet="2026").')
    if file is not None:
        if not isinstance(file, (str, Path)):
            raise PlotsimpleError("file= is the name of a file, such as \"results.xlsx\".\n"
                                  "For a table you already have, write ps.start(table=...).")
        return Story(Table(file, sheet=sheet, fast=fast))
    if example is not None:
        from .data import example as get_example

        return Story(Table(get_example(example), fast=fast))
    if isinstance(table, (str, Path)):
        raise PlotsimpleError(f'table= is a table you already have. For a file, write ps.start(file="{table}").')
    if isinstance(table, Story):
        table = table._table
    if getattr(table, "_is_plotsimple_table", False):
        return Story(Table(_frame=table._frame))  # a new table: the story never changes yours
    return Story(Table(table, fast=fast))
