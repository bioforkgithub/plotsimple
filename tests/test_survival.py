"""Tests for the survival words (Kaplan-Meier, log-rank, Cox) and chart.survival_curves.

Reference values come from R 4.6.2 with the survival package that ships with it, on
pandas and Polars tables. The R data:
  trial: d <- read.csv(".../data/trial.csv"); d$ev <- as.integer(d$died == "yes")
  ties:  the TIES table below (many tied times, 3 groups), written to CSV and read in R.
"""

import math

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.survival import kaplan_meier

ENGINES = ["pandas", "polars"]
TOL = 1e-9

TIES = pd.DataFrame({
    "time": [17, 9, 6, 7, 1, 21, 3, 4, 18, 7, 4, 19, 5, 7, 18, 6, 2, 7, 20, 1, 3, 22, 5, 6, 1, 9, 5, 7, 2, 14,
             6, 3, 24, 1, 3, 4, 13, 7, 10, 7, 6, 1, 4, 1, 15],
    "event": [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0,
              1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    "group": ["a", "b", "c"] * 15,
    "x": [43.5, 48.3, 66.6, 56.6, 33.6, 49.9, 43.8, 51.5, 33.9, 52.4, 52.4, 65.8, 53.2, 55.1, 35.1, 72.5, 30.8,
          61.0, 46.7, 41.2, 43.4, 43.3, 53.8, 48.9, 64.8, 31.7, 50.0, 41.1, 57.8, 28.8, 46.6, 52.1, 35.2, 59.9,
          51.8, 60.1, 59.6, 40.2, 42.0, 48.0, 57.5, 58.5, 42.9, 43.9, 42.0],
})


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


def frame(result):
    return result._frame.to_pandas()


def one_row(result):
    f = frame(result)
    assert len(f) == 1
    return f.iloc[0].to_dict()


@pytest.fixture
def trial():
    return sdata.example("trial")


# ---------------------------------------------------------------- survival_over_time
def test_kaplan_meier_matches_r_trial(trial, engine, capsys):
    # summary(survfit(Surv(days, ev) ~ treatment, data = d))
    out = frame(make(trial, engine).survival_over_time(time="days", event="died", for_each="treatment"))
    assert list(out.columns) == ["treatment", "time", "at risk", "events", "censored", "survival",
                                 "95 % CI low", "95 % CI high"]
    new = out[out["treatment"] == "new medicine"].reset_index(drop=True)
    assert new["time"].tolist()[:4] == [46, 50, 52, 149]
    assert new["at risk"].tolist()[:4] == [60, 59, 58, 57]
    assert new["survival"].tolist()[:8] == pytest.approx(
        [0.983333333333, 0.966666666667, 0.95, 0.933333333333, 0.916666666667, 0.9, 0.883333333333,
         0.866666666667], abs=1e-11)
    assert new["95 % CI low"].tolist()[3:8] == pytest.approx(
        [0.872303335587, 0.849333910320, 0.827204002647, 0.805727739765, 0.784783435268], abs=1e-11)
    assert new["95 % CI high"].tolist()[:5] == pytest.approx(
        [1, 1, 1, 0.998633245538, 0.989337370812], abs=1e-11)
    usual = out[out["treatment"] == "usual care"].tail(3)
    assert usual["time"].tolist() == [729, 867, 1031]
    assert usual["at risk"].tolist() == [13, 9, 2]
    assert usual["survival"].tolist() == pytest.approx([0.390062082324, 0.346721850954, 0.173360925477], abs=1e-11)
    assert usual["95 % CI low"].tolist() == pytest.approx([0.2713638096837, 0.2255167524227, 0.0406199338206],
                                                          abs=1e-11)
    assert usual["95 % CI high"].tolist() == pytest.approx([0.560680616343, 0.533069231610, 0.739883295109],
                                                           abs=1e-11)
    assert out["events"].sum() == 59
    text = capsys.readouterr().out
    # print(survfit(...)): medians 706 (486, NA) and 545 (394, NA)
    assert '"died" = "yes" (59 of 120 rows)' in text
    assert 'new medicine: median "days" 706 (95 % CI 486 to not reached; 60 rows, 25 events)' in text
    assert 'usual care: median "days" 545 (95 % CI 394 to not reached; 60 rows, 34 events)' in text
    assert 'survfit(Surv(days, died == "yes") ~ treatment)' in text


def test_kaplan_meier_one_group(trial, engine, capsys):
    out = frame(make(trial, engine).survival_over_time(time="days", event="died"))
    assert "treatment" not in out.columns
    assert 'median "days" 614 (95 % CI 526 to not reached; 120 rows, 59 events)' in capsys.readouterr().out


def test_kaplan_meier_ties_and_censored_counts(engine, capsys):
    # summary(survfit(Surv(time, event) ~ group)) rows 1, 2, 5, 9, 10
    out = frame(make(TIES, engine).survival_over_time(time="time", event="event", for_each="group"))
    rows = out.iloc[[0, 1, 4, 8, 9]]
    assert rows["time"].tolist() == [1, 3, 6, 2, 3]
    assert rows["at risk"].tolist() == [15, 13, 10, 12, 10]
    assert rows["events"].tolist() == [2, 1, 1, 2, 2]
    assert rows["censored"].tolist() == [0, 0, 1, 0, 0]
    assert rows["survival"].tolist() == pytest.approx([0.8666666666667, 0.8, 0.6, 0.6666666666667, 0.5333333333333])
    assert rows["95 % CI low"].tolist() == pytest.approx(
        [0.7106365849275, 0.6211555075328, 0.3969192926761, 0.4661238363817, 0.3322110116929], abs=1e-12)
    assert rows["95 % CI high"].tolist() == pytest.approx(
        [1, 1, 0.9069853913444, 0.9534900594109, 0.8562161831872], abs=1e-12)
    text = capsys.readouterr().out
    assert '"event" = 1 (31 of 45 rows)' in text
    assert 'a: median "time" 7 (95 % CI 5 to not reached' in text
    assert 'b: median "time" 4 (95 % CI 2 to 7' in text
    assert "c: the median was not reached" in text


def test_median_exactly_one_half_as_r(engine, capsys):
    # survfit(Surv(1:10, rep(1, 10)) ~ 1): median 5.5, 0.95LCL 3, UCL NA
    t = make(pd.DataFrame({"t": range(1, 11), "e": [True] * 10}), engine)
    t.survival_over_time(time="t", event="e")
    text = capsys.readouterr().out
    assert 'median "t" 5.5 (95 % CI 3 to not reached' in text
    assert '"e" = True' in text


def test_survival_reaching_zero_has_no_interval():
    km = kaplan_meier([1, 2, 3], [True, True, True])
    assert km["survival"].tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    assert math.isnan(km["low"][-1]) and math.isnan(km["high"][-1])


def test_empty_cells_skipped_with_note(trial, engine, capsys):
    data = trial.copy()
    data["days"] = data["days"].astype(float)
    data.loc[[0, 1], "days"] = np.nan
    data.loc[2, "died"] = None
    out = frame(make(data, engine).survival_over_time(time="days", event="died"))
    assert "Note: 3 rows with an empty cell" in capsys.readouterr().out
    assert out["events"].sum() + out["censored"].sum() <= 117


def test_event_values_read_as_yes_no(trial, engine, capsys):
    base = frame(make(trial, engine).survival_over_time(time="days", event="died"))
    for values, said in ((trial["died"].eq("yes"), "True"), (trial["died"].eq("yes").astype(int), "1"),
                         (trial["died"].str.upper(), '"YES"')):
        data = trial.assign(died=values)
        out = frame(make(data, engine).survival_over_time(time="days", event="died"))
        assert out["survival"].tolist() == pytest.approx(base["survival"].tolist())
        assert f'"died" = {said}' in capsys.readouterr().out


def test_survival_errors(trial, engine):
    t = make(trial, engine)
    with pytest.raises(SimplydataError, match="column name"):
        t.survival_over_time(time="days")
    with pytest.raises(SimplydataError, match="other than 0 and 1"):
        make(trial.assign(died=2), engine).survival_over_time(time="days", event="died")
    with pytest.raises(SimplydataError, match="should say whether"):
        t.survival_over_time(time="days", event="treatment")
    with pytest.raises(SimplydataError, match="should hold numbers"):
        t.survival_over_time(time="patient", event="died")
    with pytest.raises(SimplydataError, match="below 0"):
        make(trial.assign(days=-trial["days"]), engine).survival_over_time(time="days", event="died")
    with pytest.raises(SimplydataError, match="days_"):
        t.survival_over_time(time="days_", event="died")
    with pytest.raises(SimplydataError, match="both"):
        t.survival_over_time(time="days", event="died", for_each="died")


# ---------------------------------------------------------------- compare_survival
def test_log_rank_matches_r_trial(trial, engine, capsys):
    # survdiff(Surv(days, ev) ~ treatment, data = d)
    r = one_row(make(trial, engine).compare_survival(time="days", event="died", for_each="treatment"))
    assert r["test"] == "log-rank test"
    assert r["statistic"] == pytest.approx(1.53386475701214, rel=TOL)
    assert r["degrees of freedom"] == 1
    assert r["p-value"] == pytest.approx(0.215533339073761, rel=TOL)
    assert r["events in new medicine"] == 25 and r["events in usual care"] == 34
    assert r["expected events in new medicine"] == pytest.approx(29.7457169465, rel=1e-10)
    assert r["expected events in usual care"] == pytest.approx(29.2542830535, rel=1e-10)
    text = capsys.readouterr().out
    assert "new medicine 25 vs 29.7 and usual care 34 vs 29.3" in text
    assert "could easily be chance (p = 0.216)" in text
    assert "chi-square = 1.53, df = 1" in text


def test_log_rank_three_groups_with_ties(engine):
    # survdiff(Surv(time, event) ~ group)
    r = one_row(make(TIES, engine).compare_survival(time="time", event="event", for_each="group"))
    assert r["statistic"] == pytest.approx(15.0463429999238, rel=TOL)
    assert r["degrees of freedom"] == 2
    assert r["p-value"] == pytest.approx(0.00054041591604996, rel=1e-8)
    expected = [r[f"expected events in {g}"] for g in "abc"]
    assert expected == pytest.approx([11.05584261187709, 7.27775796741314, 12.66639942070977], rel=TOL)


def test_log_rank_errors(trial, engine):
    t = make(trial, engine)
    with pytest.raises(SimplydataError, match="for_each="):
        t.compare_survival(time="days", event="died")
    with pytest.raises(SimplydataError, match="at least 2 groups"):
        make(trial.assign(treatment="same"), engine).compare_survival(time="days", event="died", for_each="treatment")
    with pytest.raises(SimplydataError, match="no events"):
        make(trial.assign(died="no"), engine).compare_survival(time="days", event="died", for_each="treatment")


# ---------------------------------------------------------------- explain_survival
def test_cox_matches_r_trial(trial, engine, capsys):
    # coxph(Surv(days, ev) ~ treatment + age, data = d); cox.zph(fit)
    out = frame(make(trial, engine).explain_survival(time="days", event="died", using=["treatment", "age"]))
    assert out["term"].tolist() == ["treatment: usual care vs new medicine", "age"]
    assert out["coefficient"].tolist() == pytest.approx([0.2884939103543782, 0.0504026366313829], rel=1e-7)
    assert out["standard error"].tolist() == pytest.approx([0.2644477059374165, 0.0118932250703042], rel=1e-7)
    assert out["hazard ratio"].tolist() == pytest.approx([1.3344162234, 1.0516944619], rel=1e-8)
    assert out["95 % CI low"].tolist() == pytest.approx([0.794681345, 1.027462678], rel=1e-8)
    assert out["95 % CI high"].tolist() == pytest.approx([2.24073041, 1.07649773], rel=1e-8)
    assert out["p-value"].tolist() == pytest.approx([0.2753037, 2.2559e-05], rel=1e-4)
    assert out["proportional hazards check p-value"].tolist() == pytest.approx([0.578412619317, 0.114768253213],
                                                                               rel=1e-7)
    text = capsys.readouterr().out
    # likelihood-ratio test 21.643, df 2, p = 1.9965e-05
    assert "chi-square = 21.6, df = 2) is very unlikely to be chance (p < 0.0001)" in text
    assert '"treatment" usual care vs new medicine: hazard ratio 1.33 (95 % CI 0.795 to 2.24)' in text
    assert 'each 1 more "age": hazard ratio 1.05 (95 % CI 1.03 to 1.08)' in text
    assert "Efron" in text and "may change over time" not in text


def test_cox_with_ties_and_three_groups(engine, capsys):
    # coxph(Surv(time, event) ~ group + x) (Efron); cox.zph(fit); breslow would give 0.889 for groupb
    out = frame(make(TIES, engine).explain_survival(time="time", event="event", using=["group", "x"]))
    assert out["coefficient"].tolist() == pytest.approx([0.9503263116621189, -0.9290602922537614, 0.0330996498726311],
                                                        rel=1e-7)
    assert out["standard error"].tolist() == pytest.approx(
        [0.4129089048567882, 0.5421699120985614, 0.0196910896027861], rel=1e-7)
    assert out["p-value"].tolist() == pytest.approx([0.0213611338638822, 0.0866029149453946, 0.0927734870549661],
                                                    rel=1e-6)
    # cox.zph: group (2 df) 0.3321055033795, x 0.5680589921925
    assert out["proportional hazards check p-value"].tolist() == pytest.approx(
        [0.3321055033795, 0.3321055033795, 0.5680589921925], rel=1e-7)
    # summary(fit)$logtest: 17.3186877513130 on 3 df
    assert "chi-square = 17.3, df = 3" in capsys.readouterr().out


def test_cox_proportional_hazards_note(engine, capsys):
    # crossing curves: coxph(Surv(t, e) ~ g): coef -0.346573590279972; cox.zph p = 0.0008110300693533
    data = pd.DataFrame({"t": list(range(1, 9)) + [30] * 8 + list(range(20, 28)) + [30] * 8,
                         "e": (["yes"] * 8 + ["no"] * 8) * 2, "g": ["early"] * 16 + ["late"] * 16})
    out = frame(make(data, engine).explain_survival(time="t", event="e", using="g"))
    assert out["coefficient"].iloc[0] == pytest.approx(-0.346573590279972, rel=1e-7)
    assert out["proportional hazards check p-value"].iloc[0] == pytest.approx(0.0008110300693533, rel=1e-6)
    assert 'Note: the hazard ratio of "g" may change over time' in capsys.readouterr().out


def test_cox_errors(trial, engine):
    t = make(trial, engine)
    with pytest.raises(SimplydataError, match="using="):
        t.explain_survival(time="days", event="died")
    with pytest.raises(SimplydataError, match="twice"):
        t.explain_survival(time="days", event="died", using=["age", "age"])
    with pytest.raises(SimplydataError, match="same value"):
        make(trial.assign(age=50), engine).explain_survival(time="days", event="died", using="age")
    with pytest.raises(SimplydataError, match="say the same thing"):
        make(trial.assign(age2=trial["age"] * 2), engine).explain_survival(time="days", event="died",
                                                                           using=["age", "age2"])


def test_words_do_not_change_the_table(trial, engine):
    t = make(trial, engine)
    t.survival_over_time(time="days", event="died")
    t.compare_survival(time="days", event="died", for_each="treatment")
    t.explain_survival(time="days", event="died", using="age")
    pd.testing.assert_frame_equal(t._frame.to_pandas(), make(trial, engine)._frame.to_pandas())


# ---------------------------------------------------------------- chart.survival_curves
def _curves(ax):
    return [ln for ln in ax.lines if ln.get_drawstyle().startswith("steps")]


def test_chart_numbers_equal_the_table(trial, capsys):
    chart = sdata.chart(trial).survival_curves(time="days", event="died", colour="treatment")
    ax = chart._render().axes[0]
    table = frame(sdata.table(trial).survival_over_time(time="days", event="died", for_each="treatment"))
    lines = _curves(ax)
    assert [ln.get_label() for ln in lines] == ["new medicine", "usual care"]
    for line, name in zip(lines, ["new medicine", "usual care"]):
        part = table[table["treatment"] == name]
        x, y = line.get_xdata(), line.get_ydata()
        assert x[0] == 0 and y[0] == 1
        assert list(x[1:-1]) == part["time"].tolist()
        assert list(y[1:-1]) == pytest.approx(part["survival"].tolist())
        assert x[-1] == trial.loc[trial["treatment"] == name, "days"].max()
    marks = [ln for ln in ax.lines if ln.get_marker() == "|"]
    censored = trial[trial["died"] == "no"].groupby("treatment")["days"].nunique().tolist()
    assert [len(m.get_xdata()) for m in marks] == censored
    assert ax.get_ylabel() == "Survival (share without the event)"
    assert ax.get_legend().get_title().get_text() == "treatment"


def test_chart_band_risk_table_and_options(trial):
    chart = sdata.chart(trial).survival_curves(time="days", event="died", colour="treatment",
                                               confidence_band=True, censor_marks=False, at_risk_table=True)
    ax = chart._render().axes[0]
    assert len(ax.collections) == 2  # one band per group
    assert not [ln for ln in ax.lines if ln.get_marker() == "|"]
    texts = [t.get_text() for t in ax.texts]
    assert "Number at risk" in texts and texts.count("60") == 2  # everyone at risk at 0


def test_chart_panels_letters_and_empty_cells(trial, tmp_path, capsys):
    data = trial.copy()
    data.loc[0, "died"] = None
    one = sdata.chart(data).survival_curves(time="days", event="died", colour="red")
    assert _curves(one._render().axes[0])[0].get_color() == "red"
    assert "1 row with empty" in capsys.readouterr().out
    panels = sdata.chart(trial).survival_curves(time="days", event="died").one_chart_per("treatment")
    assert len(panels._render().axes) == 2
    sdata.side_by_side(one, panels, letters=True).save(tmp_path / "page.png")


def test_chart_errors(trial):
    chart = sdata.chart(trial)
    with pytest.raises(SimplydataError, match="event="):
        chart.survival_curves(time="days")
    with pytest.raises(SimplydataError, match="True or False"):
        chart.survival_curves(time="days", event="died", confidence_band="yes")
    with pytest.raises(SimplydataError, match="should say whether"):
        chart.survival_curves(time="days", event="treatment")
    with pytest.raises(SimplydataError, match="should hold numbers"):
        chart.survival_curves(time="patient", event="died")


def test_survival_curves_transparency_and_thickness():
    chart = sdata.chart(sdata.example("trial")).survival_curves(
        time="days", event="died", colour="treatment", transparency=0.4, thickness=3)
    lines = [ln for ln in chart._render().axes[0].lines if ln.get_drawstyle() == "steps-post"]
    assert len(lines) == 2
    assert all(abs(ln.get_alpha() - 0.6) < 1e-12 and ln.get_linewidth() == 3 for ln in lines)
    with pytest.raises(sdata.SimplydataError, match="transparency"):
        sdata.chart(sdata.example("trial")).survival_curves(time="days", event="died", transparency=2)
