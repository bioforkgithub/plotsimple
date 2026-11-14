"""Survival: how long until something happens, when some rows were followed only part of the way.

time= is the column with the time until the event, or until the end of follow-up for rows
where the event did not happen (they are "censored": we only know the event had not
happened by then). event= is the column that says whether the event happened: yes/no,
True/False or 1/0, and the printed sentence says which value counts as the event.

Methods (the same defaults as R's survival package, so results can be checked there):

* survival_over_time: the Kaplan-Meier estimate (R survfit(Surv(time, event) ~ group)).
  At each time with events, survival is multiplied by (1 - events / at risk); "at risk"
  counts the rows with a time at or after that moment (events come before censoring at a
  tied time, as in R). 95 % confidence intervals on the log scale (R's default
  conf.type = "log"): S x exp(-/+ 1.96 se), se of log S from Greenwood's formula
  sqrt(sum(d / (n (n - d)))), the top end cut at 1. The median time and its 95 % CI are
  read where the curve and the two edges of the band first reach 0.5 or below, with R's
  rule for a curve that sits exactly on 0.5 (the middle of that time and the next drop).
* compare_survival: the log-rank test (R survdiff(), rho = 0). At each event time the
  events are shared out over the groups in proportion to the rows at risk (the expected
  events); the chi-square compares observed and expected with their hypergeometric
  variance, on (groups - 1) degrees of freedom.
* explain_survival: the Cox proportional hazards model (R coxph()), with Efron's method for
  tied event times (R's default). Fitted here with NumPy by Newton-Raphson with
  step-halving, from all coefficients at 0, as R does (stops when the log partial
  likelihood changes by less than 1e-9 of itself; at most 20 steps), so statsmodels is not
  needed. Hazard ratios exp(coefficient) with 95 % Wald intervals, Wald z p-values, and the
  likelihood-ratio test of the whole model (2 x the gain in log partial likelihood, df =
  number of coefficients). A text (or True/False) column becomes groups compared with the
  first group in sorted order, as R does with a factor.
  Proportional-hazards check: the score test of R cox.zph() (survival 3.x) with its
  default transform = "km": for each term, does its effect change with 1 - KM(t-) (the
  left-continuous Kaplan-Meier estimate of all rows, centred on its mean over events)?
  The test uses the full information matrix of the model with the extra time terms, at
  the fitted coefficients, with Efron's weights for tied times. A note is printed when a
  term has p < 0.05.
"""

from __future__ import annotations

import math

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, friendly
from .stat_tests import _chance, _label, _list_words, _num, _p_shown, _skipped_note, _whole
from .summaries import _p_words

Z95 = 1.959963984540054  # qnorm(0.975)
EVENT_WORDS = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
EXAMPLE = 'with sdata.example("trial"): table.{word}(time="days", event="died"{more})'
COX_STEPS = 20  # R coxph.control(iter.max = 20)
COX_EPS = 1e-9  # R coxph.control(eps = 1e-9)


# ---------------------------------------------------------------------------- reading the columns
def is_empty(value) -> bool:
    """An empty cell: missing, NaN, or text that is only spaces."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def read_event(values, column: str):
    """The event column as True/False (True = the event happened), and words saying which value
    counts as the event. values: a list without empty cells."""
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return np.array([bool(v) for v in values], dtype=bool), "True"
    if all(_is_number(v) for v in values):
        found = sorted({float(v) for v in values})
        if set(found) <= {0.0, 1.0}:
            return np.array([float(v) == 1.0 for v in values], dtype=bool), "1"
        shown = ", ".join(_num(v) for v in found[:6])
        raise SimplydataError(
            f'The event column "{column}" has numbers other than 0 and 1 ({shown}).\n'
            "It should say whether the event happened: 1/0, yes/no or True/False."
        )
    if all(isinstance(v, str) for v in values):
        words = {v.strip().lower() for v in values}
        if words <= set(EVENT_WORDS):
            out = np.array([EVENT_WORDS[v.strip().lower()] for v in values], dtype=bool)
            yes = [v.strip() for v in values if EVENT_WORDS[v.strip().lower()]]
            if not yes:
                yes = ["yes" if words & {"yes", "no"} else "true" if words & {"true", "false"} else "1"]
            return out, f'"{yes[0]}"'
        other = sorted(words - set(EVENT_WORDS))[:4]
        raise SimplydataError(
            f'The event column "{column}" should say whether the event happened (yes/no, True/False or 1/0), '
            f"but it also has {_list_words(repr(o) for o in other)}.\n"
            "Change those first, for example with table.replace_values(...), or use another column."
        )
    kinds = sorted({type(v).__name__ for v in values})
    raise SimplydataError(
        f'The event column "{column}" holds a mix of things ({", ".join(kinds)}). '
        "It should say whether the event happened: yes/no, True/False or 1/0."
    )


def check_time(values, column: str):
    """The time column as floats: numbers, none below 0."""
    if not all(_is_number(v) for v in values):
        raise SimplydataError(
            f'The time column "{column}" should hold numbers (the time until the event or the end of '
            "follow-up), but it holds other things."
        )
    times = np.asarray(values, dtype=float)
    if (times < 0).any():
        raise SimplydataError(
            f'The time column "{column}" has {int((times < 0).sum())} values below 0. A time until an event '
            "cannot be negative; check those rows."
        )
    return times


def _need(step, option, value, example):
    if not isinstance(value, str):
        raise SimplydataError(f"{step}() needs {option} as a column name in quotes.\nFor example, {example}")


# ---------------------------------------------------------------------------- the calculations
def kaplan_meier(time, event) -> dict:
    """Kaplan-Meier estimate as R survfit() (conf.type = "log").

    At each event time: time, at_risk, events, censored (since the previous event time, up
    to and including this time, as R summary(survfit)), survival, se (of log survival,
    Greenwood), low, high. Also at every distinct time (all_time, all_survival, all_low,
    all_high), the censored times with their survival, and the last time (end)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    all_time, inverse = np.unique(time, return_inverse=True)
    deaths = np.bincount(inverse, weights=event.astype(float), minlength=len(all_time))
    lost = np.bincount(inverse, weights=(~event).astype(float), minlength=len(all_time))
    at_risk = (deaths + lost)[::-1].cumsum()[::-1]
    survival = np.cumprod(1 - deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(deaths > 0, deaths / (at_risk * (at_risk - deaths)), 0.0)
        se = np.sqrt(np.cumsum(step))  # infinite once survival reaches 0
        low = survival * np.exp(-Z95 * se)
        high = np.minimum(1.0, survival * np.exp(Z95 * se))
    gone = (survival == 0) | ~np.isfinite(se)
    low[gone], high[gone] = np.nan, np.nan
    keep = deaths > 0
    censored = np.diff(np.concatenate([[0.0], np.cumsum(lost)[keep]]))
    return dict(
        time=all_time[keep], at_risk=at_risk[keep], events=deaths[keep], censored=censored,
        survival=survival[keep], se=se[keep], low=low[keep], high=high[keep],
        all_time=all_time, all_at_risk=at_risk, all_survival=survival, all_low=low, all_high=high,
        censor_time=all_time[lost > 0], censor_survival=survival[lost > 0],
        end=float(time.max()) if len(time) else 0.0, n=len(time), total_events=int(event.sum()),
    )


def _first_half(values, times):
    """R survmean's minmin(): the first time the curve is at 0.5 or below (with R's tolerance);
    exactly on 0.5 gives the middle of that time and the next time it drops."""
    tolerance = math.sqrt(np.finfo(float).eps)
    keep = ~np.isnan(values) & (values < 0.5 + tolerance)
    if not keep.any():
        return math.nan
    x, y = times[keep], values[keep]
    if abs(y[0] - 0.5) < tolerance and (y < y[0]).any():
        return float((x[0] + x[np.argmax(y < y[0])]) / 2)
    return float(x[0])


def median_time(km) -> tuple[float, float, float]:
    """Median survival time with its 95 % CI, as R print(survfit()); nan = not reached."""
    t = km["all_time"]
    return (_first_half(km["all_survival"], t), _first_half(km["all_low"], t), _first_half(km["all_high"], t))


def log_rank(time, event, codes, k):
    """Log-rank test as R survdiff(rho = 0): observed, expected, chi-square, df."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    observed = np.array([(event & (codes == j)).sum() for j in range(k)], dtype=float)
    expected = np.zeros(k)
    variance = np.zeros((k, k))
    for t in np.unique(time[event]):
        risk = time >= t
        n_j = np.array([(risk & (codes == j)).sum() for j in range(k)], dtype=float)
        n = n_j.sum()
        d = float((event & (time == t)).sum())
        share = n_j / n
        expected += d * share
        if n > 1:
            variance += d * (n - d) / (n - 1) * (np.diag(share) - np.outer(share, share))
    diff = (observed - expected)[:-1]
    small = variance[:-1, :-1]
    chi = float(diff @ np.linalg.pinv(small) @ diff)
    df = int(np.linalg.matrix_rank(small))
    return observed, expected, chi, df


def _efron_sums(time, event, x, beta, gtime=None):
    """Log partial likelihood, score and information of a Cox model with Efron's ties.

    With gtime (one value per row, the transformed time of R cox.zph), it also gives the
    score of the extra terms x * g(t) and the information of [x, x * g(t)], at beta."""
    p = x.shape[1]
    eta = x @ beta
    r = np.exp(eta)
    loglik = 0.0
    score = np.zeros(p)
    info = np.zeros((p, p))
    zph_u = np.zeros(p)
    zph_i = np.zeros((2 * p, 2 * p))
    for t in np.unique(time[event]):
        risk = time >= t
        dead = event & (time == t)
        d = int(dead.sum())
        rr, xr = r[risk], x[risk]
        s0, s1, s2 = rr.sum(), rr @ xr, (xr * rr[:, None]).T @ xr
        rd, xd = r[dead], x[dead]
        d0, d1, d2 = rd.sum(), rd @ xd, (xd * rd[:, None]).T @ xd
        loglik += eta[dead].sum()
        step_u = xd.sum(axis=0)
        step_i = np.zeros((p, p))
        for j in range(d):
            f = j / d
            a0, a1, a2 = s0 - f * d0, s1 - f * d1, s2 - f * d2
            mean = a1 / a0
            loglik -= math.log(a0)
            step_u = step_u - mean
            step_i += a2 / a0 - np.outer(mean, mean)
        score += step_u
        info += step_i
        if gtime is not None:
            g = float(gtime[dead][0])
            zph_u += g * step_u
            zph_i[:p, :p] += step_i
            zph_i[:p, p:] += g * step_i
            zph_i[p:, :p] += g * step_i
            zph_i[p:, p:] += g * g * step_i
    return loglik, score, info, zph_u, zph_i


def _same_terms_error():
    return SimplydataError("The model cannot be fitted: two of the terms say the same thing "
                           "(or a term has the same value in every row with an event).")


def cox_fit(time, event, x) -> dict:
    """Cox model by Newton-Raphson with step-halving, as R coxph() (Efron ties)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0)  # as R: centring changes no result, but keeps exp() in range
    p = x.shape[1]
    beta = np.zeros(p)
    loglik0, score, info, _, _ = _efron_sums(time, event, x, beta)
    loglik = loglik0
    converged = False
    for _step in range(COX_STEPS):
        try:
            change = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            raise _same_terms_error() from None
        new = beta + change
        new_loglik, new_score, new_info, _, _ = _efron_sums(time, event, x, new)
        halvings = 0
        while (not np.isfinite(new_loglik) or new_loglik < loglik) and halvings < 30:
            new = (new + beta) / 2
            new_loglik, new_score, new_info, _, _ = _efron_sums(time, event, x, new)
            halvings += 1
        done = abs(1 - loglik / new_loglik) <= COX_EPS if new_loglik != 0 else abs(new_loglik - loglik) <= COX_EPS
        beta, loglik, score, info = new, new_loglik, new_score, new_info
        if done:
            converged = True
            break
    try:
        var = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        raise _same_terms_error() from None
    return dict(beta=beta, var=var, loglik0=loglik0, loglik=loglik, converged=converged, x=x)


def cox_zph(time, event, fit, terms) -> list[tuple[float, int, float]]:
    """R cox.zph(fit) (transform = "km"): (chi-square, df, p) for each term, then GLOBAL.
    terms: a list of column-index lists, one per term."""
    from scipy import stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    km = kaplan_meier(time, event)
    before = np.searchsorted(km["all_time"], time, side="left")  # how many times are strictly earlier
    ttimes = 1 - np.concatenate([[1.0], km["all_survival"]])[before]
    gtime = ttimes - ttimes[event].mean()
    _, _, _, u, imat = _efron_sums(time, event, fit["x"], fit["beta"], gtime)
    p = len(fit["beta"])
    out = []
    for cols in list(terms) + [list(range(p))]:
        kk = list(range(p)) + [p + c for c in cols]
        vec = np.concatenate([np.zeros(p), u[cols]])
        try:
            chi = float(np.linalg.solve(imat[np.ix_(kk, kk)], vec) @ vec)
        except np.linalg.LinAlgError:
            chi = math.nan
        out.append((chi, len(cols), float(stats.chi2.sf(chi, len(cols))) if not math.isnan(chi) else math.nan))
    return out


def _show(value) -> str:
    """A median time for a sentence: "not reached" when there is none."""
    if math.isnan(value):
        return "not reached"
    return f"{value:.6g}"


def _surv(time, event, yes) -> str:
    """The R Surv() call for a sentence: Surv(days, died == "yes") for a text event column."""
    return f"Surv({time}, {event} == {yes})" if yes.startswith('"') else f"Surv({time}, {event})"


def group_order(keys):
    """Group names A to Z (numbers by size), as the other test words sort them."""
    return sorted(set(keys), key=lambda k: (str(type(k)), k))


# ---------------------------------------------------------------------------- the words
class SurvivalWords:
    def _survival_data(self, step, time, event, extra=(), example=""):
        """time (floats), event (bools), the used columns as lists, and the value that counts as
        the event. Rows with an empty cell in any of these columns are skipped with a note."""
        _need(step, "time=", time, example)
        _need(step, "event=", event, example)
        columns = list(dict.fromkeys([time, event] + list(extra)))
        self._need_columns(*columns)
        if time == event:
            raise SimplydataError(f'"{time}" cannot be both time= and event=.\nFor example, {example}')
        values = {c: self._frame[c].to_list() for c in columns}
        n = len(self._frame)
        keep = [i for i in range(n) if not any(is_empty(values[c][i]) for c in columns)]
        _skipped_note(n - len(keep), [c for c in columns if any(is_empty(v) for v in values[c])])
        if not keep:
            raise SimplydataError(f"{step}() found no rows with {_list_words(repr(c) for c in columns)} all filled in.")
        kept = {c: [values[c][i] for i in keep] for c in columns}
        times = check_time(kept[time], time)
        events, yes = read_event(kept[event], event)
        return times, events, kept, yes

    @staticmethod
    def _event_sentence(event, yes, events, time):
        return (f'The event is "{event}" = {yes} ({int(events.sum())} of {len(events)} rows); the other rows count as '
                f'followed without the event until their "{time}" (censored).')

    @staticmethod
    def _group_codes(keys):
        order = group_order(keys)
        place = {k: i for i, k in enumerate(order)}
        return [_label(k) for k in order], np.array([place[k] for k in keys])

    def _check_for_each(self, step, for_each, time, event, example):
        _need(step, "for_each=", for_each, example)
        if for_each in (time, event):
            raise SimplydataError(f'"{for_each}" cannot be both for_each= and time= or event=.\nFor example, {example}')

    # ------------------------------------------------------------------ survival_over_time
    @friendly
    def survival_over_time(self, time=None, event=None, for_each=None):
        """The share still without the event over time (Kaplan-Meier), with the median time.

        table.survival_over_time(time="days", event="died", for_each="treatment")   (with sdata.example("trial"))
        time= is the time until the event or the end of follow-up; event= says whether the
        event happened (yes/no, True/False or 1/0). for_each= gives one curve per group.
        Prints the median time (by when half have had the event) with its 95 % CI per group,
        or says it was not reached, as R print(survfit(Surv(days, died == "yes") ~ treatment)).
        Gives back a table with one row per event time: the group, time, at risk, events,
        censored (since the previous row), survival and its 95 % CI (log scale, R's
        default), as R summary(survfit(...)). Draw the curves with chart.survival_curves(...).
        """
        step = "survival_over_time"
        example = EXAMPLE.format(word=step, more=', for_each="treatment"')
        if for_each is not None:
            self._check_for_each(step, for_each, time, event, example)
        times, events, kept, yes = self._survival_data(step, time, event, [for_each] if for_each else [], example)
        if for_each:
            names, codes = self._group_codes(kept[for_each])
        else:
            names, codes = [None], np.zeros(len(times), dtype=int)
        columns = ["time", "at risk", "events", "censored", "survival", "95 % CI low", "95 % CI high"]
        keys = ("time", "at_risk", "events", "censored", "survival", "low", "high")
        out = {c: [] for c in ([for_each] if for_each else []) + columns}
        print(self._event_sentence(event, yes, events, time))
        print("Median time until the event (by then half have had it), with its 95 % CI:")
        for j, name in enumerate(names):
            km = kaplan_meier(times[codes == j], events[codes == j])
            if for_each:
                out[for_each] += [name] * len(km["time"])
            for c, key in zip(columns, keys):
                out[c] += [int(v) if c in ("at risk", "events", "censored") else float(v) for v in km[key]]
            med, low, high = median_time(km)
            who = f"{name}: " if for_each else ""
            count = f"{km['n']} rows, {km['total_events']} events"
            if not km["total_events"]:
                print(f"  {who}no events ({count}), so the median was not reached.")
            elif math.isnan(med):
                print(f"  {who}the median was not reached ({count}): more than half were still without the event "
                      f"at the end of follow-up (survival {_num(km['survival'][-1])} after the last event).")
            else:
                print(f'  {who}median "{time}" {_show(med)} (95 % CI {_show(low)} to {_show(high)}; {count}).')
        group = for_each if for_each else "1"
        print(f"Method: Kaplan-Meier estimate with 95 % confidence intervals on the log scale (Greenwood), "
              f"as R survfit({_surv(time, event, yes)} ~ {group}).")
        return self._new_table(nw.from_dict(out, backend=self._frame.implementation))

    # ------------------------------------------------------------------ compare_survival
    @friendly
    def compare_survival(self, time=None, event=None, for_each=None):
        """Do the groups differ in the time until the event, or could it be chance? (log-rank test)

        table.compare_survival(time="days", event="died", for_each="treatment")   (with sdata.example("trial"))
        The log-rank test, as R survdiff(Surv(days, died == "yes") ~ treatment): at each event
        time the events are shared out over the groups by how many rows are at risk in each,
        and the observed events are compared with these expected ones over the whole
        follow-up. Prints the answer in words and the observed vs expected events per group;
        gives back a one-row table of results.
        """
        step = "compare_survival"
        example = EXAMPLE.format(word=step, more=', for_each="treatment"')
        self._check_for_each(step, for_each, time, event, example)
        times, events, kept, yes = self._survival_data(step, time, event, [for_each], example)
        names, codes = self._group_codes(kept[for_each])
        k = len(names)
        if k < 2:
            raise SimplydataError(f'{step}() needs at least 2 groups in "{for_each}"; it found only "{names[0]}".')
        if not events.any():
            raise SimplydataError(f'{step}() found no events ("{event}" = {yes}) in any group, so there is nothing '
                                  "to compare.")
        from scipy import stats

        observed, expected, chi, df = log_rank(times, events, codes, k)
        p = float(stats.chi2.sf(chi, df)) if df > 0 else math.nan
        print(self._event_sentence(event, yes, events, time))
        counts = _list_words(f"{n} {_whole(o)} vs {_num(e)}" for n, o, e in zip(names, observed, expected))
        print(f'Events in each "{for_each}" group, observed vs expected if all groups were alike: {counts}.')
        print(f"Differences this big in the time until the event {_chance(p, plural=True)}.")
        print(f"Test: log-rank test (chi-square = {_num(chi)}, df = {df}), as R survdiff({_surv(time, event, yes)} ~ "
              f"{for_each}); it compares the whole curves, giving every event time the same weight.")
        row = {"test": "log-rank test",
               "what it compares": f'the time until "{event}" = {yes} in each "{for_each}" ({", ".join(names)})',
               "statistic name": "chi-square", "statistic": chi, "degrees of freedom": float(df), "p-value": p}
        for j, name in enumerate(names):
            row[f"count in {name}"] = int((codes == j).sum())
            row[f"events in {name}"] = int(observed[j])
            row[f"expected events in {name}"] = float(expected[j])
        return self._result_table(row)

    # ------------------------------------------------------------------ explain_survival
    def _cox_columns(self, using, kept, n_rows):
        """The model columns: numbers as they are, text and True/False as groups vs the first."""
        columns, names, terms = [], [], []
        for column in using:
            values = kept[column]
            if all(_is_number(v) for v in values):
                values = np.asarray(values, dtype=float)
                if np.ptp(values) == 0:
                    raise SimplydataError(f'"{column}" has the same value in every row, so it cannot explain anything.')
                terms.append([len(columns)])
                columns.append(values)
                names.append((column, None, None))
                continue
            keys = [bool(v) if isinstance(v, (bool, np.bool_)) else str(v) for v in values]
            order = group_order(keys)
            if len(order) < 2:
                raise SimplydataError(f'"{column}" has only one group ("{order[0]}"), so it cannot explain anything.')
            if len(order) > max(20, n_rows // 2):
                raise SimplydataError(f'"{column}" has {len(order)} different values: too many groups to fit. '
                                      "Is it a column of numbers stored as text?")
            cols = []
            for level in order[1:]:
                cols.append(len(columns))
                columns.append(np.array([k == level for k in keys], dtype=float))
                names.append((column, _label(level), _label(order[0])))
            terms.append(cols)
        return np.column_stack(columns), names, terms

    @friendly
    def explain_survival(self, time=None, event=None, using=None):
        """Which columns go with a higher or lower risk of the event? (Cox model, hazard ratios)

        table.explain_survival(time="days", event="died", using=["treatment", "age"])   (with sdata.example("trial"))
        The Cox proportional hazards model, as R coxph(Surv(days, died == "yes") ~ treatment +
        age), with Efron's method for tied times. A hazard ratio above 1 means a higher risk
        of the event at any moment (2 = twice the risk), below 1 a lower risk. Numbers count
        per 1 more; text columns are groups compared with the first group (A to Z). Prints
        each hazard ratio with its 95 % CI and p-value, and the likelihood-ratio test of the
        whole model; a note says if a hazard ratio seems to change over time (the
        proportional-hazards check of R cox.zph()). Gives back a table with one row per term.
        """
        step = "explain_survival"
        example = EXAMPLE.format(word=step, more=', using=["treatment", "age"]')
        if isinstance(using, str):
            using = [using]
        if not using or not all(isinstance(u, str) for u in using):
            raise SimplydataError(f"{step}() needs using= with one or more column names.\nFor example, {example}")
        using = list(using)
        repeated = [u for i, u in enumerate(using) if u in using[:i]]
        if repeated:
            raise SimplydataError(f'"{repeated[0]}" is in using= twice. Give each column once.')
        if time in using or event in using:
            raise SimplydataError(f"using= cannot hold the time= or event= column.\nFor example, {example}")
        times, events, kept, yes = self._survival_data(step, time, event, using, example)
        if not events.any():
            raise SimplydataError(f'{step}() found no events ("{event}" = {yes}), so there is nothing to explain.')
        x, names, terms = self._cox_columns(using, kept, len(times))
        if x.shape[1] >= len(times):
            raise SimplydataError(f"{step}() needs more rows than terms: {len(times)} rows for {x.shape[1]} terms.")
        from scipy import stats

        fit = cox_fit(times, events, x)
        beta, se = fit["beta"], np.sqrt(np.diag(fit["var"]))
        z = beta / se
        p = 2 * stats.norm.sf(np.abs(z))
        lr = 2 * (fit["loglik"] - fit["loglik0"])
        df = len(beta)
        lr_p = float(stats.chi2.sf(lr, df))
        zph = cox_zph(times, events, fit, terms)
        term_of = {c: i for i, cols in enumerate(terms) for c in cols}
        print(self._event_sentence(event, yes, events, time))
        print(f"Hazard ratios from {len(times)} rows with {int(events.sum())} events (above 1: a higher risk of the "
              "event at any moment; below 1: a lower risk):")
        rows = []
        for i, (column, level, base) in enumerate(names):
            hr, low, high = math.exp(beta[i]), math.exp(beta[i] - Z95 * se[i]), math.exp(beta[i] + Z95 * se[i])
            what = f'each 1 more "{column}"' if level is None else f'"{column}" {level} vs {base}'
            print(f"  {what}: hazard ratio {_num(hr)} (95 % CI {_num(low)} to {_num(high)}); {_p_words(float(p[i]))}.")
            rows.append({"term": column if level is None else f"{column}: {level} vs {base}",
                         "hazard ratio": hr, "95 % CI low": low, "95 % CI high": high,
                         "coefficient": float(beta[i]), "standard error": float(se[i]), "z": float(z[i]),
                         "p-value": float(p[i]), "proportional hazards check p-value": zph[term_of[i]][2]})
        print(f"The model as a whole (likelihood-ratio test, chi-square = {_num(lr)}, df = {df}) {_chance(lr_p)}.")
        print(f"Method: Cox proportional hazards model, Efron's method for tied times, as R "
              f"coxph({_surv(time, event, yes)} ~ {' + '.join(using)}).")
        if not fit["converged"]:
            print(f"Note: the model did not settle after {COX_STEPS} steps; a hazard ratio may be heading to 0 or "
                  "infinity (for example a group with no events). Treat the numbers with care.")
        elif np.any(np.abs(beta) * np.std(fit["x"], axis=0) > 10):
            print("Note: a hazard ratio is extremely large or small; perhaps a group has no events or "
                  "all of them. Treat it with care.")
        for i, cols in enumerate(terms):
            if zph[i][2] < 0.05:
                column = names[cols[0]][0]
                print(f'Note: the hazard ratio of "{column}" may change over time (proportional-hazards check as R '
                      f"cox.zph(), {_p_shown(zph[i][2])}), so it is an average over the follow-up. "
                      "chart.survival_curves(...) shows how the curves run.")
        if zph[-1][2] < 0.05 and all(t[2] >= 0.05 for t in zph[:-1]):
            print(f"Note: the proportional-hazards check of the whole model has {_p_shown(zph[-1][2])} "
                  "(as R cox.zph()): some hazard ratio may change over time.")
        frame = nw.from_dict({key: [r[key] for r in rows] for key in rows[0]}, backend=self._frame.implementation)
        return self._new_table(frame)
