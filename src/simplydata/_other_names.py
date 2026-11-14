"""Other names: the words people bring from other tools and from everyday maths.

People choose the same name for a thing surprisingly rarely (Furnas et al. 1987, "The vocabulary
problem in human-system communication"), so a system that knows only one name fails most
first tries. simplydata keeps one word for each idea, but it knows the other names too: typing
one of them never runs anything, it gives a friendly pointer, "In simplydata this is ...".

Each list below is (how the simplydata word is written, the other names that point to it).
The tests check that every target is a real word and that no other name is a word itself.
"""

from __future__ import annotations

# sdata.<name>: top-level words, and words people expect on sdata that live on a table or a chart.
_SDATA = [
    ('sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])',
     "minimize minimise optimize optimise optim minimize_scalar fmin fminbound argmin nlm "
     "nlminb fminsearch"),
    ('sdata.find_biggest("x * (10 - x)", between=[0, 10])', "maximize maximise argmax fmax"),
    ('sdata.find_smallest("2*x + 3*y", rules=["x + y >= 4", "x >= 0", "y >= 0"])',
     "linprog lp milp linear_programming"),
    ('sdata.solve("x**3 - 2*x = 5", between=[0, 3])',
     "root roots fsolve brentq brenth bisect newton root_scalar uniroot polyroot nleqslv fzero zeros_of"),
    ('table.fit_formula("population = a * exp(b * (year - 2000))"), on a table',
     "curve_fit nls nlsLM least_squares leastsq"),
    ('table.fit_curve(x="hours", y="marks", degree=2), on a table', "polyfit"),
    ('table.fit_line(x="hours", y="marks"), on a table', "linregress lm ols"),
    ('table.values_at("population", at={"year": [2001, 2003]}), on a table',
     "interpolate interp interp1d approx approxfun spline splinefun CubicSpline pchip PchipInterpolator "
     "make_interp_spline"),
    ('table.map_of("height", across=["east", "north"]), on a table', "griddata interp2d RBFInterpolator"),
    ('sdata.distribution("normal", average=50, standard_deviation=10).density_at(60)', "pdf dnorm pmf dbinom"),
    ('sdata.distribution("normal", average=50, standard_deviation=10).chance(below=60)',
     "cdf pnorm pbinom sf"),
    ('sdata.distribution("normal", average=50, standard_deviation=10).percentile(95), or '
     '.cut_off(above=0.025) for a critical value', "ppf qnorm isf"),
    ("nums.percentile(90) for numbers, table.percentile_of(\"marks\", 90) for a column, or "
     "sdata.distribution(\"normal\").percentile(90)", "quantile percentile"),
    ('sdata.random_numbers(how_many=100, average=50, standard_deviation=10, seed=1), or '
     'sdata.distribution("gamma", shape=2).random_numbers(how_many=100)', "rvs rnorm runif rand randn random"),
    ('sdata.distribution("normal", average=50, standard_deviation=10)', "norm rv_continuous"),
    ('table.fit_distribution("days", "gamma"), on a table', "fit fitdistr fitdist fit_distribution"),
    ('sdata.table("my_file.csv")', "read_csv read_excel read_table read_parquet read_json DataFrame data_frame "
     "tibble read"),
    ("sdata.numbers(0, to=10, how_many=50)", "linspace arange seq seq_len array"),
    ("sdata.number_grid(rows=2, columns=3, filled_with=0)", "zeros ones matrix full"),
    ('sdata.chart(table), then chart.lines(x="year", y="population")', "plot figure subplots ggplot"),
    ('table.compare_averages("marks", for_each="class"), on a table',
     "ttest ttest_ind t_test anova aov f_oneway"),
    ('table.compare_averages("marks", for_each="class", based_on="ranks"), on a table',
     "mannwhitneyu wilcox_test kruskal kruskal_test"),
    ('table.compare_counts("result", for_each="class"), on a table', "chi2_contingency chisq_test fisher_exact"),
    ('table.correlation_between("hours", "marks"), on a table', "pearsonr spearmanr kendalltau cor cor_test"),
    ('table.compare_shape_to("marks", "normal"), on a table', "shapiro shapiro_test normaltest anderson"),
    ("sdata.how_many_needed(effect_size=0.5, power=0.8)", "power_t_test solve_power pwr"),
]

# table.<name>
_TABLE = [
    ('table.average_of("marks", for_each="class")', "groupby group_by aggregate agg"),
    ('table.join_with(other_table, using="name")', "merge join left_join inner_join"),
    ("table.add_rows_from(other_table)", "concat append rbind vstack bind_rows"),
    ('table.turn_rows_into_columns(names_from="subject", values_from="marks")',
     "pivot pivot_wider pivot_table spread unstack"),
    ('table.turn_columns_into_rows(["temperature", "rain"])', "melt pivot_longer gather stack unpivot"),
    ("table.show_first_rows()", "head"),
    ("table.show_last_rows()", "tail"),
    ('table.keep_rows("marks", above=50)', "loc filter query where subset"),
    ("table.keep_rows_by_position(10, to=20)", "iloc slice take gather_every nth"),
    ("table.remove_rows_by_position([12, 40])", "drop_index"),
    ('table.cell_at(row=3, column="marks")', "at iat item get_value"),
    ('table.keep_columns(kind="number")', "select_dtypes"),
    ('table.reverse_rows() to turn the table upside down, or table.sort_by("marks", biggest_first=True) for an '
     "order by a column", "reverse flip flipud"),
    ('table.replace_values("marks", above=100, with_value=100)', "mask clip"),
    ('table.keep_columns("name", "marks")', "select"),
    ('table.remove_columns("height")', "drop"),
    ('table.rename_column("marks", to="score")', "rename"),
    ('table.sort_by("marks")', "sort_values arrange order sort"),
    ("table.remove_rows_with_empty_cells()", "dropna drop_nulls na_omit"),
    ('table.fill_empty_cells("price", with_the="median")', "fillna fill_null fill_nan replace_na"),
    ('table.values_at("population", at={"year": [2001, 2003]}) for values between the known points, or '
     'table.fill_empty_cells("price", with_the="line between neighbours") to fill gaps',
     "interpolate interp interp1d approx approxfun spline splinefun CubicSpline pchip"),
    ('table.map_of("height", across=["east", "north"])', "griddata interp2d"),
    ("table.remove_duplicate_rows()", "drop_duplicates unique distinct"),
    ("table.describe_columns()", "describe info str glimpse dtypes"),
    ('table.summary_of("marks", for_each="class")', "summary summarise summarize"),
    ('table.average_of("marks")', "mean"),
    ('table.total_of("marks")', "sum"),
    ('table.smallest_of("marks")', "min minimum"),
    ('table.biggest_of("marks")', "max maximum"),
    ('table.standard_deviation_of("marks")', "std sd"),
    ('table.median_of("marks")', "median"),
    ('table.percentile_of("marks", 90)', "quantile percentile"),
    ('table.count_each("class")', "value_counts table tabulate"),
    ('table.count_different_values("name")', "nunique n_distinct n_unique"),
    ('table.most_common_of("marks")', "mode"),
    ("table.count_rows()", "shape len nrow height count"),
    ("table.column_names()", "columns names colnames"),
    ('table.make_numbers("price")', "astype as_numeric to_numeric cast"),
    ('table.make_dates("date")', "to_datetime as_date"),
    ('table.make_lowercase("name")', "lower tolower"),
    ('table.make_uppercase("name")', "upper toupper"),
    ('table.remove_extra_spaces("name")', "strip trimws str_trim"),
    ('table.round_to("price", 1)', "round"),
    ('table.save("results.csv")', "to_csv to_excel write_csv write_xlsx to_parquet write_parquet"),
    ("table.chart()", "plot ggplot"),
    ('table.add_column("marks_per_hour", "marks / hours")', "assign mutate with_columns transform"),
    ('table.replace_values("class", "A", with_value="Alpha")', "replace recode"),
    ('table.add_previous_value_column("last_year", from_column="population")', "shift lag"),
    ('table.add_change_column("growth", from_column="population")', "diff pct_change"),
    ('table.add_rolling_column("smooth", from_column="temperature")', "rolling rollmean"),
    ('table.add_rank_column("rank", from_column="marks")', "rank"),
    ('table.add_group_column("result", from_column="marks", cut_at=[60])', "cut qcut"),
    ('table.add_is_empty_column("no_price", from_column="price")', "isna isnull is_na is_null"),
    ("table.keep_random_rows(5, seed=1)", "sample sample_n slice_sample"),
    ('table.keep_biggest_rows(3, by="marks")', "nlargest top_n slice_max"),
    ('table.keep_smallest_rows(3, by="marks")', "nsmallest slice_min"),
    ('table.correlation_between("hours", "marks")', "corr cor pearsonr spearmanr"),
    ('table.fit_formula("population = a * exp(b * (year - 2000))")', "curve_fit nls nlsLM least_squares"),
    ('table.fit_curve(x="hours", y="marks", degree=2)', "polyfit"),
    ('table.fit_line(x="hours", y="marks") for one column, or table.explain("marks", using=["hours", '
     '"breakfast"]) for several', "linregress lm ols"),
    ('table.explain("passed", using=["hours"], outcome="yes or no")', "glm logit lmer"),
    ('table.fit_distribution("days", "gamma")', "fit fitdistr fitdist"),
    ('table.compare_averages("marks", for_each="class")', "ttest ttest_ind t_test anova aov f_oneway"),
    ('table.compare_before_after("Maths", "Science")', "ttest_rel paired_t_test wilcoxon"),
    ('table.compare_counts("result", for_each="class")', "chi2_contingency chisq_test fisher_exact crosstab"),
    ('table.compare_shape_to("marks", "normal")', "shapiro shapiro_test normaltest"),
    ('table.add_adjusted_p_value_column("q", method="benjamini-hochberg")', "p_adjust multipletests fdr"),
    ('table.survival_over_time(time="days", event="died")', "survfit kaplan_meier"),
    ('table.compare_survival(time="days", event="died", for_each="treatment")', "survdiff logrank_test"),
    ('table.explain_survival(time="days", event="died", using=["age"])', "coxph cox"),
    ('table.principal_components_of()', "pca prcomp"),
    ('table.add_clusters(how_many=3)', "kmeans hclust"),
    ('table.distances_between_rows(measure="bray-curtis")', "pdist vegdist"),
    ('table.add_nmds()', "metaMDS nmds"),
    ('table.compare_profiles(for_each="grazed")', "adonis adonis2 permanova"),
]

# chart.<name>
_CHART = [
    ('chart.lines(x="year", y="population")', "plot geom_line"),
    ('chart.dots(x="hours", y="marks")', "geom_point scatterplot"),
    ('chart.bars(x="class", y="marks")', "barplot geom_bar geom_col"),
    ('chart.bars(x="class", y="marks", sideways=True)', "barh"),
    ('chart.histogram(x="marks")', "hist geom_histogram histplot"),
    ('chart.boxes(x="class", y="marks")', "boxplot geom_boxplot"),
    ('chart.violins(x="class", y="marks")', "violinplot geom_violin"),
    ('chart.density(x="marks")', "kdeplot geom_density kde"),
    ('chart.cumulative_share(x="marks")', "ecdfplot ecdf stat_ecdf"),
    ("chart.fit_line()", "regplot lmplot geom_smooth"),
    ("chart.fit_curve(degree=2)", "polyfit"),
    ('chart.fit_formula("marks = a * hours ** b")', "curve_fit nls"),
    ('chart.fit_distribution("normal")', "fitdistr fitdist"),
    ('chart.curve("y = 100 * exp(-0.3 * x)", between=[0, 10])', "stat_function geom_function fplot"),
    ("chart.smooth_line()", "lowess loess"),
    ('chart.qq_dots("marks")', "probplot qqplot qqnorm"),
    ("chart.line_with_slope(1, through=(0, 0))", "abline axline"),
    ("chart.line_at_y(50)", "axhline geom_hline hline"),
    ("chart.line_at_x(2010)", "axvline geom_vline vline"),
    ("chart.shade_between_y(40, 60)", "axhspan"),
    ("chart.shade_between_x(6, 8)", "axvspan"),
    ('chart.band(x="month", between=["lowest", "highest"])', "fill_between geom_ribbon"),
    ('chart.error_bars("standard error")', "errorbar geom_errorbar"),
    ('chart.note("Cherria grows fastest", x=2020, y=12)', "text annotate"),
    ('chart.title_top("Marks in each class")', "title suptitle set_title ggtitle"),
    ('chart.label_x("Hours of study")', "xlabel set_xlabel"),
    ('chart.label_y("Marks")', "ylabel set_ylabel"),
    ("chart.range_x(0, 10)", "xlim set_xlim"),
    ("chart.range_y(0, 100)", "ylim set_ylim"),
    ("chart.log_scale_x()", "semilogx xscale"),
    ("chart.log_scale_y()", "semilogy yscale"),
    ("chart.ticks_x(1, 4, 7, 10)", "xticks set_xticks"),
    ("chart.ticks_y(0, 25, 50, 75)", "yticks set_yticks"),
    ("chart.legend_right()", "legend"),
    ("chart.show_grid()", "grid"),
    ('chart.save("chart.png")', "savefig ggsave"),
    ('chart.heatmap(x="month", y="city", value="rain")', "imshow pcolormesh geom_tile"),
    ('chart.contour(x="east", y="north", z="height")', "contourf"),
    ('chart.surface(x="east", y="north", z="height")', "plot_surface"),
    ('chart.second_y("rain")', "twinx"),
    ('chart.one_chart_per("class")', "facet_wrap facet_grid FacetGrid subplots"),
    ("chart.side_histograms()", "jointplot"),
    ('chart.all_dots(x="class", y="marks")', "stripplot swarmplot geom_jitter"),
    ('chart.count_bars(x="fruit")', "countplot"),
    ('chart.colour_scale("magma")', "cmap colormap scale_fill_viridis_c"),
    ('chart.size(width=17, height=8, unit="cm")', "figsize set_size_inches"),
]

# nums.<name> (numbers). Older ones (mean, sum, min ...) are in numbers/_base.py HINTS.
_NUMBERS = [
    ('nums.fit_distribution("gamma")', "fit fitdistr fitdist"),
    ("nums.chart()", "plot hist"),
    ("nums.keep_random_numbers(3, seed=1)", "sample choice"),
    ("nums.count_each()", "value_counts tabulate"),
]


def _build(groups):
    table = {}
    for written, names in groups:
        for name in names.split():
            assert name not in table, f"other name {name} is listed twice"
            table[name] = written
    return table


OTHER_NAMES = {"sdata": _build(_SDATA), "table": _build(_TABLE), "chart": _build(_CHART),
               "numbers": _build(_NUMBERS)}
