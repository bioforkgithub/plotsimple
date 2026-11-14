"""Choosing rows and columns by position, one row or one cell, and turning the table upside down.

Positions count from 1, both ends included, as the printed row numbers do (table.show()):

    table.keep_rows_by_position(10, to=20)       rows 10 to 20 (11 rows)
    table.keep_rows_by_position(1, step=10)      every 10th row: 1, 11, 21 ...
    table.keep_rows_by_position([1, 5, 9])       these rows, in this order
    table.remove_rows_by_position([12, 40])      the rows show_rows_with_outliers() pointed to
    table.keep_columns_by_position(1, to=3)      the first three columns
    table.row(3)                                 one row, as a dict (the table does not change)
    table.cell_at(row=3, column="marks")         one cell (the table does not change)
    table.reverse_rows()                         the last row first

Positions are of the table as it is now: after any step the rows are numbered again from 1.
Every step copies (narwhals slicing and gathering), so there are no views and no
SettingWithCopy surprises; undo() takes each step back.
"""

from __future__ import annotations

from .._errors import SimplydataError, friendly, option_error
from .._positions import read_positions
from .columns import _plain_cell
from .conditions import _rows
from .rows import COLUMN_OPTIONS, CONDITIONS


def _columns_word(n: int) -> str:
    return f"{n} column" if n == 1 else f"{n} columns"


class PositionWords:
    # ================================================================== helpers
    def _no_conditions(self, step: str, positions, conditions: dict, what: str = "rows"):
        """keep_rows_by_position("marks", above=50) by mistake: point to the word that chooses by value."""
        if not conditions:
            return
        key = next(iter(conditions))
        verb = "keep" if step.startswith("keep") else "remove"
        if what == "rows" and key in CONDITIONS:
            column = positions if isinstance(positions, str) else "marks"
            raise SimplydataError(
                f".{step}() chooses rows by their place in the table, not by their values.\n"
                f'For rows whose values meet a condition write table.{verb}_rows("{column}", {key}=...).\n'
                'For example:  table.keep_rows("marks", above=50)   (with sdata.example("marks"))'
            )
        if what == "columns" and key in COLUMN_OPTIONS:
            raise SimplydataError(
                f".{step}() chooses columns by their place in the table. To choose them by name or kind "
                f"write table.{verb}_columns({key}=...).\n"
                'For example:  table.keep_columns(kind="number")   (with sdata.example("marks"))'
            )
        raise option_error(step, key, ["positions", "to", "step"])

    def _row_hint(self, value):
        """An extra line when a column name is given where a row position goes."""
        if isinstance(value, str) and value in self._columns:
            return (f'"{value}" is a column. For rows whose {value} meets a condition use '
                    f'table.keep_rows("{value}", above=50); for the column itself, table.keep_columns("{value}").')
        return None

    def _column_hint(self, value):
        if isinstance(value, str):
            if value in self._columns:
                return f'Columns can also be chosen by name: table.keep_columns("{value}").'
            return 'Columns can also be chosen by name: table.keep_columns("name", "marks").'
        return None

    def _rows_picked(self, step: str, positions, to, step_size, example: str):
        return read_positions(
            step, positions, to, step_size, len(self._frame), noun="row", count_noun="rows",
            owner="The table has", example=example, last_hint="table.keep_last_rows(3)",
            after_hint="to turn the table upside down use table.reverse_rows()", other_hint=self._row_hint,
        )

    def _columns_picked(self, step: str, positions, to, step_size, example: str):
        return read_positions(
            step, positions, to, step_size, len(self._columns), noun="column", count_noun="columns",
            owner="The table has", example=example, last_hint='keep_columns_by_position("last")',
            after_hint="for another order list them, like [3, 2, 1]", other_hint=self._column_hint,
        )

    # ================================================================== rows by position
    @friendly
    def keep_rows_by_position(self, positions=None, to=None, step=None, **conditions):
        """Keep rows by their position, counting from 1: a run, every n-th row, or a list.

        table.keep_rows_by_position(10, to=20)   (with sdata.example("marks"))
        Both ends are included, so rows 10 to 20 are 11 rows. to="last" goes to the end;
        step=10 keeps every 10th row (1, 11, 21 ...); a list such as [1, 5, 9] keeps those rows
        in that order. The numbers are the ones table.show() prints, as pandas df.iloc[9:20].
        """
        example = 'table.keep_rows_by_position(10, to=20)   (with sdata.example("marks"))'
        self._no_conditions("keep_rows_by_position", positions, conditions)
        picked = self._rows_picked("keep_rows_by_position", positions, to, step, example)
        total = len(self._frame)
        if isinstance(picked.indices, range):
            r = picked.indices
            new = self._frame[r.start:r.stop:r.step]
        else:
            new = self._frame[picked.array().tolist()]
        print(f"Kept {picked.words()}{picked.in_order()} ({len(new)} of {_rows(total)}).")
        if picked.note:
            print(picked.note.format(verb="kept"))
        return self._change(f"keep {picked.short_words()}", new)

    @friendly
    def remove_rows_by_position(self, positions=None, to=None, step=None, **conditions):
        """Remove rows by their position, counting from 1: a run, every n-th row, or a list.

        table.remove_rows_by_position([12, 40])   (with sdata.example("marks"))
        The opposite of keep_rows_by_position, with the same positions: remove_rows_by_position(1)
        removes the first row (a second header read as data), (2, step=2) every 2nd row.
        """
        example = 'table.remove_rows_by_position([12, 40])   (with sdata.example("marks"))'
        self._no_conditions("remove_rows_by_position", positions, conditions)
        picked = self._rows_picked("remove_rows_by_position", positions, to, step, example)
        total = len(self._frame)
        if len(picked) == total:
            raise SimplydataError("That would remove every row. Keep at least one.\nFor example:  " + example)
        new = self._frame[picked.kept_positions(total).tolist()]
        print(f"Removed {picked.words()} ({len(picked)} of {_rows(total)}). {_rows(len(new))} left.")
        if picked.note:
            print(picked.note.format(verb="removed"))
        return self._change(f"remove {picked.short_words()}", new)

    # ================================================================== columns by position
    @friendly
    def keep_columns_by_position(self, positions=None, to=None, step=None, **conditions):
        """Keep columns by their position, counting from 1 at the left; it says which columns they were.

        table.keep_columns_by_position(1, to=3)   (with sdata.example("marks"))
        Both ends are included. "last" is the last column; step=2 keeps every 2nd column; a list
        such as [3, 1] keeps those columns in that order. As pandas df.iloc[:, 0:3].
        """
        example = 'table.keep_columns_by_position(1, to=3)   (with sdata.example("marks"))'
        self._no_conditions("keep_columns_by_position", positions, conditions, what="columns")
        picked = self._columns_picked("keep_columns_by_position", positions, to, step, example)
        names = [self._columns[i] for i in picked.indices]
        print(f"Kept {picked.words()}: {', '.join(names)}.")
        if picked.note:
            print(picked.note.format(verb="kept"))
        return self._change(f"keep {picked.short_words()} ({', '.join(names)})", self._frame.select(names))

    @friendly
    def remove_columns_by_position(self, positions=None, to=None, step=None, **conditions):
        """Remove columns by their position, counting from 1 at the left; it says which columns they were.

        table.remove_columns_by_position(1)   (with sdata.example("marks"))
        Handy for the "Unnamed: 0" column a CSV file often brings. Same positions as
        keep_columns_by_position.
        """
        example = 'table.remove_columns_by_position(1)   (with sdata.example("marks"))'
        self._no_conditions("remove_columns_by_position", positions, conditions, what="columns")
        picked = self._columns_picked("remove_columns_by_position", positions, to, step, example)
        if len(picked) == len(self._columns):
            raise SimplydataError("That would remove every column. Keep at least one.\nFor example:  " + example)
        names = [self._columns[i] for i in picked.indices]
        left = len(self._columns) - len(names)
        print(f"Removed {picked.words()}: {', '.join(names)}. {_columns_word(left)} left.")
        if picked.note:
            print(picked.note.format(verb="removed"))
        return self._change(f"remove {picked.short_words()} ({', '.join(names)})", self._frame.drop(names))

    # ================================================================== looking at one row or cell
    def _one_row(self, step: str, position, example: str) -> int:
        """A row position (from 1, or "last"), checked; gives it counting from 0."""
        if isinstance(position, (list, tuple, range)):
            raise SimplydataError(
                f".{step}() looks at one row. For several rows use "
                f"table.keep_rows_by_position({list(position)!r}).\nFor example:  {example}"
            )
        picked = read_positions(step, position, None, None, len(self._frame), noun="row", count_noun="rows",
                                owner="The table has", example=example, last_hint='table.row("last")',
                                other_hint=self._row_hint)
        return picked.indices[0]

    @friendly
    def row(self, position=None):
        """One row as a plain dict of column: value, counting from 1. table.row(3)

        table.row(3)   (with sdata.example("marks"))
        table.row("last") is the last row. Empty cells become None. The table does not change.
        As pandas df.iloc[2].to_dict() and Polars df.row(2, named=True).
        """
        example = 'table.row(3)   (with sdata.example("marks"))'
        if position is None:
            raise SimplydataError('.row() needs which row, counting from 1, like table.row(3) or table.row("last").'
                                  f"\nFor example:  {example}")
        i = self._one_row("row", position, example)
        values = self._frame.row(i)
        return {name: _plain_cell(value) for name, value in zip(self._columns, values)}

    @friendly
    def cell_at(self, row=None, column=None):
        """The value in one cell: a row counting from 1 and a column by name. table.cell_at(row=3, column="marks")

        table.cell_at(row=3, column="marks")   (with sdata.example("marks"))
        row="last" is the last row. An empty cell gives None, with a note. The table does not
        change. As pandas df.at[2, "marks"] and Polars df.item(2, "marks").
        """
        example = 'table.cell_at(row=3, column="marks")   (with sdata.example("marks"))'
        if row is None or column is None:
            missing = " and ".join(o for o, v in (("row=", row), ("column=", column)) if v is None)
            raise SimplydataError(f".cell_at() needs {missing}: a row counting from 1 and a column name.\n"
                                  f"For example:  {example}")
        if not isinstance(column, str):
            raise SimplydataError(
                f'column= takes the column\'s name, like column="marks" (you gave {column!r}). '
                "For a column by position use table.keep_columns_by_position(2).\n"
                f"For example:  {example}"
            )
        self._need_columns(column)
        i = self._one_row("cell_at", row, example)
        value = _plain_cell(self._frame.item(i, column))
        if value is None:
            print(f'Row {i + 1}, column "{column}" is empty.')
        return value

    # ================================================================== order
    @friendly
    def reverse_rows(self):
        """Turn the table upside down: the last row comes first. table.reverse_rows()

        table.reverse_rows()   (with sdata.example("marks"))
        For an order by the values in a column use table.sort_by("marks", biggest_first=True).
        """
        total = len(self._frame)
        print(f"Reversed the order of the {_rows(total)}: the last row is now first.")
        return self._change("reverse the order of the rows", self._frame[::-1])

