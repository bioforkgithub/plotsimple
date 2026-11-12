import pandas as pd
import pytest

import plotsimple as ps
from plotsimple.table import table as _table
ps.table = _table  # not exported until lessons cover tables
from plotsimple import PlotsimpleError


@pytest.fixture(params=["pandas", "polars"])
def table(request, marks):
    if request.param == "polars":
        pl = pytest.importorskip("polars")
        return ps.table(pl.from_pandas(marks))
    return ps.table(marks)


def test_engine_follows_input(marks):
    assert ps.table(marks)._engine == "pandas"
    pl = pytest.importorskip("polars")
    assert ps.table(pl.from_pandas(marks))._engine == "Polars"
    assert ps.table(marks, big_data=True)._engine == "Polars"


def test_done_gives_plain_table(table, marks):
    native = table.done()
    assert len(native) == len(marks)
    assert list(native.columns) == list(marks.columns)


def test_undo_steps_start_again(table, capsys):
    import narwhals as nw
    n = len(table)
    table._change("remove first row", table._frame.tail(n - 1))
    table._change("remove another row", table._frame.tail(n - 2))
    assert len(table) == n - 2
    table.steps()
    out = capsys.readouterr().out
    assert "1. remove first row" in out and "2. remove another row" in out
    table.undo()
    assert len(table) == n - 1
    table.start_again()
    assert len(table) == n
    table.undo()  # undoes "start again"
    assert len(table) == n - 1


def test_undo_with_nothing(table, capsys):
    table.undo()
    assert "Nothing to undo" in capsys.readouterr().out


def test_save_types(table, tmp_path):
    for kind in ("csv", "tsv", "xlsx", "parquet", "json"):
        table.save(tmp_path / f"t.{kind}")
        assert (tmp_path / f"t.{kind}").stat().st_size > 0
    again = ps.table(tmp_path / "t.xlsx")
    assert len(again) == len(table)
    with pytest.raises(PlotsimpleError, match=".xlsx"):
        table.save(tmp_path / "t")


def test_chart_from_table(table):
    chart = table.chart().bars(x="name", y="marks")
    assert len(chart._render().axes[0].patches) == 4
    assert len(ps.chart(table)._data) == len(table)


def test_friendly_errors(table):
    with pytest.raises(PlotsimpleError, match="no step called"):
        table.remove_emptyrows()
    with pytest.raises(PlotsimpleError, match="brackets"):
        table.title = "x"
    with pytest.raises(PlotsimpleError, match="needs some data"):
        ps.table()


def test_column_names_and_repr(table, marks):
    assert table.column_names() == list(marks.columns)
    assert "rows" in repr(table)
