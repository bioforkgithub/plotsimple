"""Other names: names from other tools (minimize, curve_fit, groupby, hist ...) point to the
simplydata word with "In simplydata this is ...", and never run."""

from __future__ import annotations

import pytest

import simplydata as sdata
from simplydata._errors import MissingWordError, other_name_target
from simplydata._other_names import OTHER_NAMES
from simplydata.chart import Chart
from simplydata.numbers.numbers import NumberGrid, Numbers
from simplydata.table import Table

OBJECTS = {"table": Table, "chart": Chart, "numbers": Numbers}
PREFIX = {"table": "table", "chart": "chart", "numbers": "nums"}


def _target_exists(written: str) -> bool:
    word = other_name_target(written)
    for prefix, owner in (("sdata.", sdata), ("table.", Table), ("chart.", Chart), ("nums.", Numbers)):
        if written.startswith(prefix):
            return word in owner.__dict__ or word in dir(owner) and not word.startswith("_")
    return False


@pytest.mark.parametrize("kind", list(OTHER_NAMES))
def test_every_pointer_names_a_real_word(kind):
    for name, written in OTHER_NAMES[kind].items():
        assert _target_exists(written), f"{kind}.{name} points to {written}, which is not a simplydata word"


@pytest.mark.parametrize("kind", list(OBJECTS))
def test_no_other_name_is_a_word_itself(kind):
    cls = OBJECTS[kind]
    words = set(cls._word_names())
    for name in OTHER_NAMES[kind]:
        assert name not in words, f"{kind}.{name} is a real word, so it cannot be an other name"
        assert name not in cls._OLD_WORDS, f"{kind}.{name} is an old name"


def test_no_other_name_is_an_sdata_word():
    for name in OTHER_NAMES["sdata"]:
        assert name not in vars(sdata), f"sdata.{name} exists, so it cannot be an other name"


@pytest.mark.parametrize("name, word", [
    ("minimize", "find_smallest"), ("minimise", "find_smallest"), ("optimize", "find_smallest"),
    ("optimise", "find_smallest"), ("argmin", "find_smallest"), ("maximize", "find_biggest"),
    ("root", "solve"), ("fsolve", "solve"), ("brentq", "solve"), ("uniroot", "solve"),
    ("curve_fit", "fit_formula"), ("nls", "fit_formula"), ("polyfit", "fit_curve"), ("linregress", "fit_line"),
    ("lm", "fit_line"), ("interpolate", "values_at"), ("interp", "values_at"), ("approx", "values_at"),
    ("spline", "values_at"), ("griddata", "map_of"), ("pdf", "distribution"), ("dnorm", "distribution"),
    ("cdf", "distribution"), ("pnorm", "distribution"), ("ppf", "distribution"), ("qnorm", "distribution"),
    ("quantile", "percentile"), ("rvs", "random_numbers"), ("rnorm", "random_numbers"), ("read_csv", "table"),
])
def test_sdata_other_names_point_to_the_word_and_never_run(name, word):
    with pytest.raises(MissingWordError) as info:
        getattr(sdata, name)
    text = str(info.value)
    assert text.startswith(f"simplydata has no word called sdata.{name}().")
    assert "\nIn simplydata this is " in text and f"{word}(" in text
    assert not hasattr(sdata, name)  # an AttributeError, so Python tools keep working


def test_distribution_pointers_name_the_distribution_words():
    for name, word in (("pdf", ".density_at(60)"), ("cdf", ".chance(below=60)"), ("ppf", ".percentile(95)"),
                       ("qnorm", ".cut_off(above=0.025)")):
        with pytest.raises(MissingWordError, match=r"sdata\.distribution\(\"normal\".*" + word.replace("(", r"\(")
                           .replace(")", r"\)").replace(".", r"\.")):
            getattr(sdata, name)


@pytest.mark.parametrize("name, word", [
    ("groupby", "average_of"), ("merge", "join_with"), ("concat", "add_rows_from"),
    ("pivot", "turn_rows_into_columns"), ("melt", "turn_columns_into_rows"), ("head", "show_first_rows"),
    ("tail", "show_last_rows"), ("iloc", "keep_rows_by_position"), ("filter", "keep_rows"), ("dropna", "remove_rows_with_empty_cells"),
    ("fillna", "fill_empty_cells"), ("sort_values", "sort_by"), ("mean", "average_of"), ("plot", "chart"),
    ("curve_fit", "fit_formula"), ("polyfit", "fit_curve"), ("linregress", "fit_line"), ("interpolate", "values_at"),
    ("griddata", "map_of"), ("fitdistr", "fit_distribution"), ("to_csv", "save"), ("value_counts", "count_each"),
    ("slice", "keep_rows_by_position"), ("take", "keep_rows_by_position"), ("gather_every", "keep_rows_by_position"),
    ("drop_index", "remove_rows_by_position"), ("at", "cell_at"), ("iat", "cell_at"), ("item", "cell_at"),
    ("select_dtypes", "keep_columns"), ("reverse", "reverse_rows"), ("flip", "reverse_rows"),
    ("mask", "replace_values"), ("clip", "replace_values"),
])
def test_table_other_names(name, word):
    table = sdata.table(sdata.example("marks"))
    with pytest.raises(MissingWordError) as info:
        getattr(table, name)
    text = str(info.value)
    assert text.startswith(f"simplydata has no step called .{name}().")
    assert f"In simplydata this is table.{word}(" in text
    assert f'sdata.help("{word}")' in text


def test_a_table_interpolate_points_to_both_words():
    with pytest.raises(MissingWordError, match="values_at.*fill_empty_cells"):
        sdata.table(sdata.example("growth")).interpolate()


@pytest.mark.parametrize("name, word", [
    ("plot", "lines"), ("hist", "histogram"), ("boxplot", "boxes"), ("kdeplot", "density"), ("ecdf", "cumulative_share"),
    ("axhline", "line_at_y"), ("axvline", "line_at_x"), ("xlabel", "label_x"), ("title", "title_top"),
    ("xlim", "range_x"), ("savefig", "save"), ("curve_fit", "fit_formula"), ("polyfit", "fit_curve"),
    ("probplot", "qq_dots"), ("lowess", "smooth_line"), ("abline", "line_with_slope"), ("legend", "legend_right"),
    ("grid", "show_grid"), ("facet_wrap", "one_chart_per"), ("stat_function", "curve"),
])
def test_chart_other_names(name, word):
    with pytest.raises(MissingWordError, match=rf"In simplydata this is chart\.{word}\("):
        getattr(sdata.chart(), name)


def test_number_other_names_and_grids():
    with pytest.raises(MissingWordError, match=r"In simplydata this is nums\.fit_distribution\("):
        sdata.numbers([1, 2, 3]).fit
    with pytest.raises(MissingWordError, match=r"In simplydata this is grid\.chart\(\)"):
        sdata.number_grid([[1, 2], [3, 4]]).plot  # the grid's own name
    with pytest.raises(MissingWordError) as info:
        sdata.number_grid([[1, 2], [3, 4]]).fit  # grids cannot fit a distribution: no pointer
    assert "In simplydata" not in str(info.value)
    with pytest.raises(MissingWordError, match=r"Did you mean nums\.average\(\)"):
        sdata.numbers([1, 2]).mean  # the older hints still work


def test_unknown_sdata_names_get_did_you_mean():
    with pytest.raises(MissingWordError, match=r"Did you mean sdata\.find_smallest\(\)"):
        sdata.find_smalest
    with pytest.raises(AttributeError):
        sdata.__not_there__
    assert hasattr(sdata, "__version__") and hasattr(sdata, "__path__")


def test_help_explains_other_names(capsys):
    sdata.help("minimize")
    out = capsys.readouterr().out
    assert out.startswith('simplydata has no word called "minimize": it is a name from other tools.')
    assert 'In simplydata this is sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5]).' in out
    assert 'sdata.help("find_smallest")' in out
    sdata.help("plot")  # a name with a pointer on sdata, on tables and on charts
    out = capsys.readouterr().out
    assert "  table.plot: In simplydata this is table.chart()." in out
    assert '  chart.plot: In simplydata this is chart.lines(x="year", y="population").' in out
    sdata.help("table.groupby")
    assert "table.average_of(" in capsys.readouterr().out


def test_grid_numbers_keep_their_own_words():
    assert isinstance(sdata.number_grid([[1, 2]]), NumberGrid)
