"""Reading more: several files at once, web addresses, reading options, Word/SPSS/Stata, table.show()."""

import functools
import http.server
import sys
import threading

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata import _data
from simplydata.table._base import BaseTable


@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def fast(request):
    if request.param:
        pytest.importorskip("polars")
    return request.param


def as_pandas(table):
    return table.as_pandas()


def engine_of(fast):
    return "Polars" if fast else "pandas"


# ---------------------------------------------------------------- several files
@pytest.fixture
def folder(tmp_path, marks):
    data = tmp_path / "data"
    data.mkdir()
    # Natural order: 2 before 10.
    marks.iloc[:8].to_csv(data / "class_2.csv", index=False)
    marks.iloc[8:16].to_csv(data / "class_10.csv", index=False)
    marks.iloc[16:].to_csv(data / "class_1.csv", index=False)
    return data


def test_pattern_stacks_files_with_file_column(folder, marks, fast, capsys):
    table = sdata.table(str(folder / "*.csv"), fast=fast)
    out = capsys.readouterr().out
    assert "Read 3 files and stacked them: 24 rows" in out and '"file"' in out
    assert table._engine == engine_of(fast)
    got = as_pandas(table)
    assert list(got.columns) == ["file", *marks.columns]
    assert list(dict.fromkeys(got["file"])) == ["class_1.csv", "class_2.csv", "class_10.csv"]
    expected = pd.concat([marks.iloc[16:], marks.iloc[:8], marks.iloc[8:16]], ignore_index=True)
    pd.testing.assert_frame_equal(got.drop(columns="file"), expected, check_dtype=False)


def test_folder_stacks_every_readable_file(folder, marks, fast):
    (folder / "notes.md").write_text("not data")
    (folder / "~$lock.csv").write_text("x")  # Excel lock files are left out
    table = sdata.table(folder, fast=fast)
    got = as_pandas(table)
    assert len(got) == len(marks)
    assert set(got["file"]) == {"class_1.csv", "class_2.csv", "class_10.csv"}
    assert sdata.table(str(folder) + "/", fast=fast)._engine == engine_of(fast)


def test_stacked_table_has_undo(folder, fast):
    table = sdata.table(str(folder / "*.csv"), fast=fast)
    n = len(table)
    table.sort_by("marks")
    table.undo()
    assert as_pandas(table)["file"].iloc[0] == "class_1.csv" and len(table) == n
    table.start_again()
    assert len(table) == n


def test_different_columns_give_friendly_error(folder, marks):
    marks.drop(columns="height").assign(weight=1).to_csv(folder / "class_3.csv", index=False)
    with pytest.raises(SimplydataError) as problem:
        sdata.table(folder)
    text = str(problem.value)
    assert 'class_3.csv has no column "height"' in text
    assert 'class_3.csv has an extra column "weight"' in text
    assert "Columns in class_1.csv" in text and "/*.csv" in text


def test_column_order_may_differ(folder, marks, fast):
    marks[list(reversed(marks.columns))].to_csv(folder / "class_20.csv", index=False)
    got = as_pandas(sdata.table(folder, fast=fast))
    assert list(got.columns) == ["file", *marks.columns]
    assert len(got) == 2 * len(marks)


def test_numbers_and_text_mixed_become_text(tmp_path, fast, capsys):
    (tmp_path / "a.csv").write_text("site,depth\nX,1\nY,2\n")
    (tmp_path / "b.csv").write_text("site,depth\nZ,deep\n")
    got = as_pandas(sdata.table(str(tmp_path / "*.csv"), fast=fast))
    assert "holds numbers in some files and text in others" in capsys.readouterr().out
    assert got["depth"].tolist() == ["1", "2", "deep"]


def test_existing_file_column_is_kept(tmp_path, capsys):
    (tmp_path / "a.csv").write_text("file,x\nq,1\n")
    (tmp_path / "b.csv").write_text("file,x\nr,2\n")
    got = as_pandas(sdata.table(tmp_path))
    assert list(got.columns) == ["from_file", "file", "x"]
    assert '"from_file"' in capsys.readouterr().out


def test_subfolders_named_in_file_column(tmp_path):
    for year in ("2025", "2026"):
        (tmp_path / year).mkdir()
        (tmp_path / year / "rain.csv").write_text("day,rain\n1,2.5\n")
    got = as_pandas(sdata.table(str(tmp_path / "**" / "*.csv")))
    assert got["file"].tolist() == ["2025/rain.csv", "2026/rain.csv"]


def test_no_matching_files(tmp_path, folder):
    with pytest.raises(SimplydataError, match="No files match") as problem:
        sdata.table(str(folder / "*.xlsx"))
    assert "class_1.csv" in str(problem.value)
    with pytest.raises(SimplydataError, match="does not exist"):
        sdata.table(str(tmp_path / "nowhere" / "*.csv"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SimplydataError, match="no files simplydata can read"):
        sdata.table(empty)


# ---------------------------------------------------------------- Excel: every sheet
@pytest.fixture
def workbook(tmp_path, marks):
    path = tmp_path / "lab.xlsx"
    with pd.ExcelWriter(path) as writer:
        marks.iloc[:12].to_excel(writer, sheet_name="2025", index=False)
        marks.iloc[12:].to_excel(writer, sheet_name="2026", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="notes", index=False)
    return path


def test_all_sheets_stacked(workbook, marks, fast, capsys):
    table = sdata.table(workbook, sheet="all", fast=fast)
    out = capsys.readouterr().out
    assert 'Sheet "notes" is empty' in out and "Read 2 sheets" in out
    assert table._engine == engine_of(fast)
    got = as_pandas(table)
    assert list(got.columns) == ["sheet", *marks.columns]
    assert got["sheet"].tolist() == ["2025"] * 12 + ["2026"] * 12
    pd.testing.assert_frame_equal(got.drop(columns="sheet"), marks, check_dtype=False)


def test_all_sheets_of_several_files(tmp_path, workbook, marks):
    import shutil

    workbook.rename(tmp_path / "a.xlsx")
    shutil.copy(tmp_path / "a.xlsx", tmp_path / "b.xlsx")
    got = as_pandas(sdata.table(str(tmp_path / "*.xlsx"), sheet="all"))
    assert list(got.columns[:2]) == ["file", "sheet"]
    assert len(got) == 2 * len(marks)


def test_sheets_with_different_columns(tmp_path, marks):
    path = tmp_path / "mixed.xlsx"
    with pd.ExcelWriter(path) as writer:
        marks.to_excel(writer, sheet_name="one", index=False)
        marks.drop(columns="test").to_excel(writer, sheet_name="two", index=False)
    with pytest.raises(SimplydataError, match='sheet "two" has no column "test"') as problem:
        sdata.table(path, sheet="all")
    assert 'sheet="one"' in str(problem.value)


def test_wrong_sheet_mentions_all(workbook):
    with pytest.raises(SimplydataError, match='sheet="all"'):
        sdata.table(workbook, sheet="2027")


# ---------------------------------------------------------------- reading options
EUROPEAN = "Notes from the lab\nexported 2026\nsite;rain;count\nBergen;3,5;1.200\nOslo;-;15\nTromsø;0,25;NA\n"


def test_european_csv(tmp_path):
    path = tmp_path / "lab.csv"
    path.write_text(EUROPEAN, encoding="utf-8")
    expected = pd.read_csv(path, sep=";", decimal=",", thousands=".", na_values=["NA", "-"], skiprows=2)
    got = as_pandas(sdata.table(path, separator=";", decimal_mark=",", thousands_mark=".",
                                empty_values=["NA", "-"], skip_rows=2))
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    assert got["rain"].tolist()[0] == 3.5 and got["count"].tolist()[:2] == [1200, 15]
    # The separator is found on its own too.
    again = as_pandas(sdata.table(path, decimal_mark=",", thousands_mark=".", empty_values=["NA", "-"], skip_rows=2))
    pd.testing.assert_frame_equal(again, expected, check_dtype=False)


def test_european_csv_fast_engine(tmp_path):
    pl = pytest.importorskip("polars")
    path = tmp_path / "lab.csv"
    path.write_text("site;rain\nBergen;3,5\nOslo;-\n", encoding="utf-8")
    table = sdata.table(path, decimal_mark=",", empty_values="-", fast=True)
    assert table._engine == "Polars"
    expected = pl.read_csv(path, separator=";", decimal_comma=True, null_values=["-"])
    assert table.as_polars().equals(expected)
    # thousands_mark is read by pandas and handed to Polars.
    path.write_text("site;n\nBergen;1.200\n", encoding="utf-8")
    table = sdata.table(path, decimal_mark=",", thousands_mark=".", fast=True)
    assert table._engine == "Polars" and table.as_pandas()["n"].tolist() == [1200]


def test_tab_separator_and_encoding(tmp_path, fast):
    path = tmp_path / "old.txt"
    path.write_bytes("place\tvalue\nTromsø\t1\n".encode("latin-1"))
    with pytest.raises(SimplydataError, match='encoding="latin-1"'):
        sdata.table(path, separator="tab")
    got = as_pandas(sdata.table(path, separator="tab", encoding="latin-1", fast=fast))
    assert got["place"].tolist() == ["Tromsø"]


def test_excel_reading_options(tmp_path):
    path = tmp_path / "lab.xlsx"
    pd.DataFrame({"a": ["title", "site", "Bergen", "Oslo"], "b": [None, "rain", "1", "-"]}).to_excel(
        path, index=False, header=False)
    got = as_pandas(sdata.table(path, skip_rows=1, empty_values="-"))
    expected = pd.read_excel(path, skiprows=1, na_values=["-"])
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    assert got["site"].tolist() == ["Bergen", "Oslo"] and pd.isna(got["rain"].iloc[1])


@pytest.mark.parametrize("options, message", [
    ({"decimal_mark": ";"}, 'decimal_mark= should be "." or ","'),
    ({"thousands_mark": "."}, "same as the decimal mark"),
    ({"thousands_mark": "abc"}, "one character"),
    ({"empty_values": [None]}, "empty_values="),
    ({"skip_rows": -1}, "skip_rows="),
    ({"skip_rows": 1.5}, "skip_rows="),
    ({"separator": ""}, "separator="),
    ({"encoding": "klingon"}, 'does not know the encoding "klingon"'),
    ({"table_number": 0}, "table_number="),
])
def test_reading_option_errors(tmp_path, options, message):
    path = tmp_path / "x.csv"
    path.write_text("a\n1\n")
    with pytest.raises(SimplydataError, match=message):
        sdata.table(path, **options)


def test_reading_option_for_wrong_file(tmp_path, marks):
    marks.to_parquet(tmp_path / "m.parquet")
    with pytest.raises(SimplydataError, match="skip_rows= is for CSV"):
        sdata.table(tmp_path / "m.parquet", skip_rows=1)
    with pytest.raises(SimplydataError, match="already a table"):
        sdata.table(marks, decimal_mark=",")
    with pytest.raises(SimplydataError, match="no option called"):
        sdata.table(tmp_path / "m.parquet", decimal=",")


def test_reading_options_for_charts(tmp_path):
    path = tmp_path / "lab.csv"
    path.write_text("site;rain\nBergen;3,5\n")
    data = _data.load(path, options=_data.reading_options(decimal_mark=","))
    assert data["rain"].tolist() == [3.5]


def test_parser_problem_is_friendly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('a,b\n1,"2\n')
    with pytest.raises(SimplydataError, match="could not split"):
        sdata.table(path, separator=",")


# ---------------------------------------------------------------- web addresses
@pytest.fixture
def website(tmp_path, marks):
    site = tmp_path / "site"
    site.mkdir()
    marks.to_csv(site / "marks.csv", index=False)
    marks.to_excel(site / "marks.xlsx", index=False)
    marks.to_parquet(site / "marks.parquet")
    marks.to_json(site / "marks.json", orient="records")
    (site / "export").write_text(marks.to_csv(index=False))

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("name", ["marks.csv", "marks.xlsx", "marks.parquet", "marks.json"])
def test_read_from_web_address(website, marks, name, fast, capsys):
    table = sdata.table(f"{website}/{name}", fast=fast)
    assert "Downloaded " + name in capsys.readouterr().out
    assert table._engine == engine_of(fast)
    pd.testing.assert_frame_equal(as_pandas(table), marks, check_dtype=False)


def test_web_address_without_file_type(website, marks, capsys):
    got = as_pandas(sdata.table(f"{website}/export"))
    assert "reads it as CSV text" in capsys.readouterr().out
    pd.testing.assert_frame_equal(got, marks, check_dtype=False)
    got = as_pandas(sdata.table(f"{website}/export?format=csv"))
    assert len(got) == len(marks)


def test_web_address_not_found(website):
    with pytest.raises(SimplydataError, match="404"):
        sdata.table(f"{website}/nothing.csv")


def test_offline_is_friendly(monkeypatch):
    import urllib.error
    import urllib.request

    def offline(*args, **kwargs):
        raise urllib.error.URLError("no internet")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(SimplydataError, match="Check that you are online"):
        sdata.table("https://example.org/data.csv")


# ---------------------------------------------------------------- Word, SPSS, Stata
def write_docx(path, tables):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Table 1. Marks")
    for frame in tables:
        word_table = document.add_table(rows=len(frame) + 1, cols=len(frame.columns))
        for j, name in enumerate(frame.columns):
            word_table.cell(0, j).text = str(name)
        for i, row in enumerate(frame.itertuples(index=False), 1):
            for j, value in enumerate(row):
                word_table.cell(i, j).text = str(value)
        document.add_paragraph("")
    document.save(path)


def test_word_tables(tmp_path, marks, fast, capsys):
    path = tmp_path / "report.docx"
    small = marks.iloc[:5]
    write_docx(path, [small, small.drop(columns="height"), small])
    table = sdata.table(path, fast=fast)
    assert '"report.docx" has 3 tables; reading table 1' in capsys.readouterr().out
    assert table._engine == engine_of(fast)
    pd.testing.assert_frame_equal(as_pandas(table), small.reset_index(drop=True), check_dtype=False)
    second = as_pandas(sdata.table(path, table_number=2))
    assert list(second.columns) == [c for c in marks.columns if c != "height"]
    with pytest.raises(SimplydataError, match="has 3 tables, so there is no table 4"):
        sdata.table(path, table_number=4)
    with pytest.raises(SimplydataError, match="table 2 has no column"):
        sdata.table(path, table_number="all")


def test_word_tables_all_and_decimal_comma(tmp_path):
    path = tmp_path / "eu.docx"
    frame = pd.DataFrame({"site": ["Bergen", "Oslo"], "rain": ["3,5", "-"]})
    write_docx(path, [frame, frame])
    got = as_pandas(sdata.table(path, table_number="all", decimal_mark=",", empty_values="-"))
    assert got["table"].tolist() == [1, 1, 2, 2]
    assert got["rain"].iloc[0] == 3.5 and pd.isna(got["rain"].iloc[1])


def test_word_without_tables(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("No tables here.")
    document.save(tmp_path / "plain.docx")
    with pytest.raises(SimplydataError, match="has no tables"):
        sdata.table(tmp_path / "plain.docx")


def test_spss_and_stata(tmp_path, marks, fast):
    pyreadstat = pytest.importorskip("pyreadstat")
    pyreadstat.write_sav(marks, str(tmp_path / "survey.sav"))
    table = sdata.table(tmp_path / "survey.sav", fast=fast)
    assert table._engine == engine_of(fast)
    pd.testing.assert_frame_equal(as_pandas(table), pd.read_spss(tmp_path / "survey.sav"), check_dtype=False)
    marks = marks.rename(columns={"class": "group"})  # "class" is not a Stata name
    marks.to_stata(tmp_path / "survey.dta", write_index=False)
    got = as_pandas(sdata.table(tmp_path / "survey.dta", fast=fast))
    pd.testing.assert_frame_equal(got, pd.read_stata(tmp_path / "survey.dta"), check_dtype=False)
    assert got["marks"].tolist() == marks["marks"].tolist()
    with pytest.raises(SimplydataError, match="is for CSV"):
        sdata.table(tmp_path / "survey.dta", empty_values="-")


def test_install_hints(tmp_path, monkeypatch):
    (tmp_path / "a.docx").write_bytes(b"x")
    (tmp_path / "a.sav").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "docx", None)
    monkeypatch.setitem(sys.modules, "pyreadstat", None)
    with pytest.raises(SimplydataError, match=r"pip install simplydata\[files\]"):
        sdata.table(tmp_path / "a.docx")
    with pytest.raises(SimplydataError, match=r"needs pyreadstat"):
        sdata.table(tmp_path / "a.sav")


# ---------------------------------------------------------------- table.show()
# These call BaseTable.show (the display word in table/_base.py) directly, so they test it
# even while another word group (rows.py) also defines a show() that comes first.
def show(table, *args, **kwargs):
    return BaseTable.show(table, *args, **kwargs)


@pytest.fixture(params=["pandas", "polars"])
def marks_table(request, marks):
    if request.param == "polars":
        pl = pytest.importorskip("polars")
        return sdata.table(pl.from_pandas(marks))
    return sdata.table(marks)


def test_tables_have_show(marks_table):
    assert callable(marks_table.show)


def test_show_whole_table(marks_table, marks, capsys):
    assert show(marks_table) is None
    out = capsys.readouterr().out
    assert "24 rows, 5 columns." in out
    assert "…" not in out and "..." not in out  # nothing hidden
    for name in marks["name"].unique():
        assert out.count(name) == (marks["name"] == name).sum()


def test_show_some_rows(marks_table, capsys):
    show(marks_table, rows=3)
    out = capsys.readouterr().out
    assert "Showing 3 of 24 rows, 5 columns." in out and "Dina" not in out
    show(marks_table, rows=100)
    assert "24 rows, 5 columns." in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="whole number"):
        show(marks_table, rows="ten")
    with pytest.raises(SimplydataError, match="no option called"):
        show(marks_table, row=3)


def test_show_big_table(capsys):
    table = sdata.table({"x": np.arange(1500)})
    show(table)
    out = capsys.readouterr().out
    assert "This table has 1500 rows; showing the first 1000" in out
    assert "Showing 1000 of 1500 rows" in out
    assert "999" in out and "1000  1000" not in out


def test_example_table_shows(capsys):
    show(sdata.table(sdata.example("marks")), rows=2)
    assert "Showing 2 of 54 rows" in capsys.readouterr().out
