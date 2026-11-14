"""Choosing by position, shared by tables, numbers and grids.

Positions count from 1, both ends included, the way people say "rows 10 to 20" (11 rows):

    keep_rows_by_position(3)               one row
    keep_rows_by_position(10, to=20)       a run, both ends included
    keep_rows_by_position(10, to="last")   to the end, without knowing how many there are
    keep_rows_by_position(1, step=10)      every 10th row: 1, 11, 21 ...
    keep_rows_by_position([1, 5, 9])       exactly these, in this order

"last" is a position everywhere. Negative numbers, 0, and a start after to= are friendly
errors: simplydata never counts from 0 and never counts down silently.
"""

from __future__ import annotations

import numbers as _nb

import numpy as np

from ._errors import SimplydataError

SHOW_AT_MOST = 10  # a list of positions longer than this is printed as its first 10 and "..."


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st ..."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    ending = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{ending}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _listed(values) -> str:
    values = list(values)
    shown = ", ".join(str(v) for v in values[:SHOW_AT_MOST])
    return shown + (", ..." if len(values) > SHOW_AT_MOST else "")


class Picked:
    """The positions that were asked for, counted from 0 for Python, and how to say them."""

    def __init__(self, kind, indices, noun, start=None, end=None, step=None, note=""):
        self.kind = kind  # "one", "run", "every" or "list"
        self.indices = indices  # a range or a list, counting from 0
        self.noun = noun
        self.start, self.end, self.step = start, end, step  # counting from 1, for the sentence
        self.note = note  # "... so rows 50 to 54 were {verb}." when to= went past the end

    def __len__(self):
        return len(self.indices)

    def array(self) -> np.ndarray:
        return np.asarray(list(self.indices) if isinstance(self.indices, range) else self.indices, dtype=np.int64)

    def words(self) -> str:
        """rows 10 to 20 / row 3 / every 10th row from row 1 to row 54: rows 1, 11 ... / rows 1, 5, 9"""
        noun = self.noun
        if self.kind == "one":
            return f"{noun} {self.start}"
        if self.kind == "run":
            return f"{noun}s {self.start} to {self.end}"
        listed = [i + 1 for i in self.indices]
        if self.kind == "every":
            found = f"{noun} {listed[0]}" if len(listed) == 1 else f"{noun}s {_listed(listed)}"
            return f"every {ordinal(self.step)} {noun} from {noun} {self.start} to {noun} {self.end}: {found}"
        return f"{noun} {listed[0]}" if len(listed) == 1 else f"{noun}s {_listed(listed)}"

    def short_words(self) -> str:
        """For steps(): the same without the list of every n-th position."""
        if self.kind == "every":
            return f"every {ordinal(self.step)} {self.noun} from {self.noun} {self.start} to {self.noun} {self.end}"
        return self.words()

    def in_order(self) -> str:
        """", in that order" for a list of two or more."""
        return ", in that order" if self.kind == "list" and len(self.indices) > 1 else ""

    def kept_positions(self, total: int) -> np.ndarray:
        """The positions (from 0) that are left when these are removed."""
        mask = np.ones(total, dtype=bool)
        mask[self.array()] = False
        return np.flatnonzero(mask)


def read_positions(word: str, positions, to, step, total: int, *, noun: str = "row", count_noun: str = "rows",
                   owner: str = "The table has", example: str = "", last_hint: str = "",
                   after_hint: str = "", other_hint=None, name: str = "table") -> Picked:
    """Check the positions given to a _by_position word and give them back as a Picked.

    word is the step's name (for messages), total how many rows/columns/numbers there are now.
    noun is what a position is called in sentences ("row", "column", "position"), count_noun the
    plural of what is counted ("rows", "numbers"). other_hint(value) may give an extra line for a
    wrong value (such as a column name given to keep_rows_by_position).
    """
    ex = f"\nFor example:  {example}" if example else ""
    call = f"{name}.{word}"

    def wrong(value, what=None):
        shown = repr(value)
        lines = [f"{(what or noun).capitalize()} positions are whole numbers from 1 to {total}, or \"last\" "
                 f"(you gave {shown})."]
        extra = other_hint(value) if other_hint else None
        if extra:
            lines.append(extra)
        return SimplydataError("\n".join(lines) + ex)

    def one(value, label=None):
        """One position, checked, counting from 1."""
        if isinstance(value, str):
            if value.strip().lower() == "last":
                return total
            raise wrong(value)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, _nb.Integral):
            raise wrong(value)
        value = int(value)
        if value == 0:
            raise SimplydataError(
                f"Positions count from 1, so the first {noun} is 1, not 0 "
                f"(Python's [ ] counts from 0; simplydata does not)." + ex
            )
        if value < 0:
            tail = f' or {last_hint}' if last_hint else ""
            raise SimplydataError(
                f'{value} is not a position. For the last {noun}s use to="last"{tail}; '
                f'"last" on its own is the last {noun}.' + ex
            )
        if value > total and label != "to":
            raise SimplydataError(
                f"There is no {noun} {value}: positions count from 1, so 1 is the first {noun} and "
                f"{total} is the last." + ex
            )
        return value

    if positions is None:
        raise SimplydataError(f".{word}() needs which {noun}s, like 10, to=20 or [1, 5, 9]." + ex)
    if total == 0:
        raise SimplydataError(f"There are no {count_noun} to choose from: {owner.lower()} none.")

    many = isinstance(positions, (list, tuple, range, np.ndarray, set))
    if many:
        if to is not None or step is not None:
            raise SimplydataError(
                f"to= and step= go with one starting {noun}, not with a list. Write either "
                f"{call}(10, to=20) or {call}([1, 5, 9])." + ex
            )
        if isinstance(positions, set):
            raise SimplydataError(
                f"Give the {noun}s as a list in square brackets, like [1, 5, 9]: a set has no order." + ex
            )
        values = list(positions.ravel()) if isinstance(positions, np.ndarray) else list(positions)
        if not values:
            raise SimplydataError(f"The list of {noun}s is empty. Write at least one, like [1, 5, 9]." + ex)
        if isinstance(positions, range) and positions.start == 0 and len(positions):
            raise SimplydataError(
                f"range() starts at 0, but positions count from 1, so the first {noun} is 1. Write "
                f"range(1, {positions.stop + 1}) or {call}(1, to={positions.stop})." + ex
            )
        if len(values) > total:
            raise SimplydataError(
                f"The list has {len(values)} positions, but there are only {_plural(total, noun)}. "
                f"Each {noun} can be chosen once." + ex
            )
        checked = [one(v) for v in values]
        seen = set()
        for v in checked:
            if v in seen:
                raise SimplydataError(
                    f"{noun.capitalize()} {v} is listed twice. Each {noun} can be chosen once." + ex
                )
            seen.add(v)
        if len(checked) == 1:
            return Picked("one", [checked[0] - 1], noun, start=checked[0])
        return Picked("list", [v - 1 for v in checked], noun)

    start = one(positions)
    if step is not None:
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, _nb.Integral) or step < 1:
            raise SimplydataError(
                f"step= is how many {noun}s to move on each time: a whole number from 1 up, like step=2 "
                f"(every 2nd {noun}). Positions only count up." + ex
            )
        step = int(step)
    if to is None:
        if step is None:
            return Picked("one", range(start - 1, start), noun, start=start)
        to = total
    end = one(to, label="to")
    note = ""
    if end > total:
        note = f"{owner} {_plural(total, noun if noun != 'position' else 'number')}, so {{words}} were {{verb}}."
        end = total
    if start > end:
        hint = f"; {after_hint}" if after_hint else ""
        raise SimplydataError(
            f"{noun.capitalize()} {start} comes after {noun} {end}. Write the smaller first: "
            f"{call}({end}, to={start}){hint}." + ex
        )
    if step is None or step == 1:
        kind = "one" if start == end else "run"
        picked = Picked(kind, range(start - 1, end), noun, start=start, end=end, step=None)
    else:
        picked = Picked("every", range(start - 1, end, step), noun, start=start, end=end, step=step)
    if note:
        picked.note = note.replace("{words}", picked.words() if picked.kind != "every" else
                                   f"{noun}s {start} to {end}")
    return picked
