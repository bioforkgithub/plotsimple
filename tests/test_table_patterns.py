"""Tests for the pattern words (principal components, clusters, distances, NMDS, PERMANOVA).

Reference values come from R 4.6.2 with vegan 2.7.6, on the practice tables read with
read.csv(".../data/meadow.csv") and so on; the R call is written next to each value.
sp <- meadow[, 4:13] is the ten plant-count columns. Every test runs on pandas and Polars.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import procrustes

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.patterns import _isotonic, _pava, _sum_squares

ENGINES = ["pandas", "polars"]
SPECIES = ["clover", "daisy", "buttercup", "dandelion", "nettle", "thistle", "moss", "fern", "orchid", "yarrow"]
SCHOOL_NUMBERS = ["hours", "sleep", "marks", "days_absent"]


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


@pytest.fixture
def meadow(engine):
    return make(sdata.example("meadow"), engine)


@pytest.fixture
def school(engine):
    return make(sdata.example("school"), engine)


def row(result):
    frame = result._frame.to_pandas()
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


def aligned(found, expected):
    """found with each column's sign flipped to match expected (component signs are arbitrary)."""
    found, expected = np.asarray(found, float), np.asarray(expected, float)
    return found * np.sign((found * expected).sum(axis=0))


def complete_school():
    data = sdata.example("school")
    return data, data[SCHOOL_NUMBERS].notna().all(axis=1).to_numpy()


# ---------------------------------------------------------------------------- principal components
# R: sc <- na.omit(school[, c("hours", "sleep", "marks", "days_absent")]); p <- prcomp(sc, scale. = TRUE)
SCHOOL_SDEV = [1.409857376998, 0.973018335945, 0.860875007999, 0.569589075598]
SCHOOL_ROTATION = [[0.597411468059, 0.35274327408248, -0.1040641638001, -0.712630598721],
                   [0.282051659342, -0.93187376632032, -0.0823155288322, -0.212796378774],
                   [0.572393453972, 0.08439402276687, -0.5503248490511, 0.601985002544],
                   [-0.485707868109, -0.00781761822978, -0.8243405631428, -0.290670582721]]
SCHOOL_SCORES = [[-0.597579417455, -0.382762907455], [0.628342806799, -1.251530593055],
                 [-0.124662436227, -0.265600998467]]  # head(p$x[, 1:2], 3): rows 1, 3, 4


def test_add_principal_components_matches_prcomp(school, capsys):
    out = school.add_principal_components()
    assert out is school
    text = capsys.readouterr().out
    assert "19 rows with an empty cell" in text
    assert "PC1 49.7 % and PC2 23.7 % of the spread (73.4 % together)" in text
    assert '"hours" (0.60), "marks" (0.57) and "days_absent" (-0.49)' in text
    assert "prcomp(scale. = TRUE)" in text and "arbitrary" in text
    data, kept = complete_school()
    frame = school._frame.to_pandas()
    scores = frame.loc[kept, ["PC1", "PC2"]].to_numpy()
    assert frame.loc[~kept, ["PC1", "PC2"]].isna().all().all()
    assert np.allclose(aligned(scores[:3], SCHOOL_SCORES), SCHOOL_SCORES, atol=1e-9)
    # the same signs as R here (same LAPACK routine)
    assert np.allclose(scores[:3], SCHOOL_SCORES, atol=1e-9)
    school.undo()
    assert "PC1" not in school.column_names()


def test_principal_components_of_matches_prcomp(school):
    result = school.principal_components_of()._frame.to_pandas()
    assert result["component"].tolist() == ["PC1", "PC2", "PC3", "PC4"]
    assert np.allclose(result["standard deviation"], SCHOOL_SDEV, rtol=1e-9)
    # summary(p)$importance: 0.49692, 0.23669, 0.18528, 0.08111
    assert np.allclose(result["spread explained (%)"], [49.692446, 23.669117, 18.527644, 8.110793], atol=1e-5)
    assert result["cumulative (%)"].iloc[-1] == pytest.approx(100)
    weights = result[[f"weight of {c}" for c in SCHOOL_NUMBERS]].to_numpy().T
    assert np.allclose(aligned(weights, SCHOOL_ROTATION), SCHOOL_ROTATION, atol=1e-9)
    assert school._history == []


def test_principal_components_unscaled_meadow(meadow):
    # R: p2 <- prcomp(sp); summary(p2)$importance[2, 1:3]; p2$rotation[, 1:2]
    result = meadow.principal_components_of(same_scale=False)._frame.to_pandas()
    assert np.allclose(result["spread explained (%)"][:3] / 100, [0.34497, 0.20609, 0.14096], atol=5e-6)
    assert result["standard deviation"][0] == pytest.approx(14.86811403577, rel=1e-9)
    pc1 = result.loc[0, [f"weight of {c}" for c in SPECIES]].to_numpy(float)
    r_pc1 = [-0.8271508878995, -0.3449181399172, -0.0819602927332, 0.0699717022637, 0.2080486312061,
             0.2706280361714, 0.1436135377688, 0.0627112741436, 0.0711350016111, -0.1977317607326]
    assert np.allclose(pc1 * np.sign(pc1 @ r_pc1), r_pc1, atol=1e-9)
    meadow.add_principal_components(same_scale=False, how_many=1)
    first = meadow._frame.to_pandas()["PC1"].to_numpy()
    assert abs(first[1]) == pytest.approx(33.86814662307, rel=1e-9)  # p2$x[2, 1]


def test_principal_components_errors(school, meadow):
    with pytest.raises(SimplydataError, match="how_many= as a whole number from 1 to 4"):
        school.add_principal_components(how_many=5)
    with pytest.raises(SimplydataError, match="at least 2 columns"):
        school.add_principal_components(columns=["hours"])
    with pytest.raises(SimplydataError, match="needs a column of numbers"):
        school.add_principal_components(columns=["hours", "school"])
    with pytest.raises(SimplydataError, match="not found"):
        school.add_principal_components(columns=["hours", "slep"])
    flat = make(sdata.example("meadow").assign(orchid=3), "pandas")
    with pytest.raises(SimplydataError, match="same value in every row"):
        flat.add_principal_components()
    assert school._history == []


# ---------------------------------------------------------------------------- clusters
def test_kmeans_matches_r_meadow(meadow, capsys):
    # R: set.seed(1); k1 <- kmeans(scale(sp), 3, nstart = 25)
    meadow.add_clusters()
    text = capsys.readouterr().out
    labels = meadow._frame.to_pandas()["cluster"].tolist()
    assert labels == [1, 1, 1, 2, 1, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1, 3, 2, 3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
    x = sdata.example("meadow")[SPECIES].to_numpy(float)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    within, total = _sum_squares(z, np.array(labels))
    assert within == pytest.approx(188.442099934031, rel=1e-9)  # k1$tot.withinss
    assert 1 - within / total == pytest.approx(0.350199655399892, rel=1e-9)  # betweenss / totss
    assert "sizes 12, 4 and 14" in text and "35.0 % of the spread lies between the clusters" in text
    assert "Lloyd" in text and "Hartigan" in text
    meadow.undo()
    assert "cluster" not in meadow.column_names()


@pytest.mark.parametrize("k, same_scale, r_within", [
    (4, True, 165.206452687565),     # set.seed(1); kmeans(scale(sp), 4, nstart = 25)$tot.withinss
    (3, False, 11056.0952380952),    # set.seed(1); kmeans(sp, 3, nstart = 25)$tot.withinss
])
def test_kmeans_within_sum_of_squares(meadow, k, same_scale, r_within):
    meadow.add_clusters(how_many=k, same_scale=same_scale)
    x = sdata.example("meadow")[SPECIES].to_numpy(float)
    z = (x - x.mean(0)) / x.std(0, ddof=1) if same_scale else x - x.mean(0)
    labels = meadow._frame.to_pandas()["cluster"].to_numpy()
    assert _sum_squares(z, labels)[0] == pytest.approx(r_within, rel=1e-9)


def test_kmeans_school_with_empty_cells(school, capsys):
    # R: sc <- na.omit(...); set.seed(1); k2 <- kmeans(scale(sc), 3, nstart = 25)
    school.add_clusters()
    text = capsys.readouterr().out
    assert "19 rows with an empty cell" in text
    data, kept = complete_school()
    labels = school._frame.to_pandas()["cluster"]
    assert labels[~kept].isna().all()
    x = data.loc[kept, SCHOOL_NUMBERS].to_numpy(float)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    within, total = _sum_squares(z, labels[kept].to_numpy(int))
    assert within == pytest.approx(367.120649923442, rel=1e-9)
    assert 1 - within / total == pytest.approx(0.426373984494622, rel=1e-9)
    assert sorted(labels[kept].value_counts().tolist()) == [41, 46, 74]
    assert "42.6 %" in text


# R: cutree(hclust(dist(scale(sp)), method = m), 3)
HCLUST = {
    "ward": [1, 1, 1, 2, 1, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    "average": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "complete": [1, 2, 2, 1, 2, 3, 3, 3, 3, 3, 1, 2, 2, 1, 2, 3, 1, 3, 3, 3, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3],
    "single": [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


@pytest.mark.parametrize("linkage", list(HCLUST))
def test_hierarchical_matches_hclust(meadow, linkage, capsys):
    meadow.add_clusters(method="hierarchical", linkage=linkage, name="group")
    assert meadow._frame.to_pandas()["group"].tolist() == HCLUST[linkage]
    assert "hclust" in capsys.readouterr().out


def test_hierarchical_school_ward(school):
    # R: head(cutree(hclust(dist(scale(sc)), method = "ward.D2"), 3), 20)
    school.add_clusters(method="hierarchical")
    _, kept = complete_school()
    labels = school._frame.to_pandas()["cluster"][kept].to_numpy(int)
    assert labels[:20].tolist() == [1, 1, 1, 2, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    assert sorted(np.bincount(labels)[1:].tolist()) == [22, 23, 116]


def test_clusters_leave_out_earlier_pattern_columns(meadow, capsys):
    meadow.add_clusters()
    meadow.add_principal_components()
    capsys.readouterr()
    meadow.add_clusters(method="hierarchical", name="h")
    text = capsys.readouterr().out
    assert "left out cluster, PC1 and PC2" in text and "from 10 columns" in text
    assert meadow._frame.to_pandas()["h"].tolist() == HCLUST["ward"]
    assert len(meadow._history) == 3


def test_cluster_errors(meadow):
    with pytest.raises(SimplydataError, match="k-means"):
        meadow.add_clusters(method="kmeans")
    with pytest.raises(SimplydataError, match="ward"):
        meadow.add_clusters(method="hierarchical", linkage="wards")
    with pytest.raises(SimplydataError, match="how_many="):
        meadow.add_clusters(how_many=1)
    with pytest.raises(SimplydataError, match="no option called"):
        meadow.add_clusters(clusters=3)


# ---------------------------------------------------------------------------- distances
@pytest.mark.parametrize("measure, expected", [
    # R: as.matrix(vegdist(sp, m))[c(1, 9, 17), c(2, 20, 30)]
    ("bray-curtis", [[0.336492890995261, 0.354497354497354, 0.351351351351351],
                     [0.461538461538462, 0.294117647058824, 0.367521367521368],
                     [0.548076923076923, 0.376344086021505, 0.483516483516484]]),
    ("euclidean", [[38.9486841883009, 26.0576284415908, 28.7923600977759],
                   [50.0199960015992, 26.8700576850888, 33.0454232837166],
                   [59.2114853723498, 27.5680975041804, 33.5857112474933]]),
    ("manhattan", [[71, 67, 65], [120, 70, 86], [114, 70, 88]]),
    ("jaccard", [[0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0]]),  # vegdist(sp, "jaccard", binary = TRUE)
])
def test_distances_match_vegdist(meadow, measure, expected):
    result = meadow.distances_between_rows(measure=measure, names="plot")._frame.to_pandas()
    assert result.columns[0] == "plot" and result.shape == (30, 31)
    picked = result.set_index("plot").iloc[[0, 8, 16], [1, 19, 29]].to_numpy()
    assert np.allclose(picked, expected, rtol=1e-12, atol=1e-12)
    square = result.set_index("plot").to_numpy()
    assert np.allclose(square, square.T) and np.all(np.diag(square) == 0)


def test_distances_jaccard_is_presence_absence(engine):
    data = pd.DataFrame({"site": ["a", "b", "c"], "x": [5, 1, 0], "y": [0, 2, 3], "z": [1, 0, 4]})
    result = make(data, engine).distances_between_rows(measure="jaccard", names="site")._frame.to_pandas()
    # a = {x, z}, b = {x, y}, c = {y, z}: 2 of the 3 species found in either are in only one
    assert result.set_index("site").loc["a", "b"] == pytest.approx(2 / 3)


def test_distances_without_names_and_with_empty_cells(engine, capsys):
    data = sdata.example("meadow")
    data.loc[2, "moss"] = None
    result = make(data, engine).distances_between_rows()._frame.to_pandas()
    assert "1 row with an empty cell" in capsys.readouterr().out
    assert result.columns[0] == "row" and result["row"].tolist()[:3] == ["1", "2", "4"]
    assert result.shape == (29, 30)


def test_distance_errors(engine):
    data = pd.DataFrame({"site": ["a", "a", "c"], "x": [5, -1, 0], "y": [0, 2, 3]})
    t = make(data, engine)
    with pytest.raises(SimplydataError, match="cannot be below 0"):
        t.distances_between_rows()
    t2 = make(pd.DataFrame({"x": [1, 0, 3], "y": [2, 0, 1]}), engine)
    with pytest.raises(SimplydataError, match="nothing in it"):
        t2.distances_between_rows()
    t3 = make(pd.DataFrame({"site": ["a", "a", "c"], "x": [1, 2, 3]}), engine)
    with pytest.raises(SimplydataError, match="more than once"):
        t3.distances_between_rows(names="site")
    with pytest.raises(SimplydataError, match="bray-curtis"):
        t3.distances_between_rows(measure="bray")


# ---------------------------------------------------------------------------- NMDS
# R: set.seed(1); m1 <- metaMDS(sp, distance = "bray", k = 2, autotransform = FALSE, trymax = 100)
R_NMDS_BRAY = [(-0.078615, -0.049263), (-0.302451, -0.395405), (-0.412042, -0.140473), (0.187378, 0.383363),
               (-0.276229, -0.192488), (0.332471, -0.013189), (0.311597, -0.198325), (0.497046, 0.122559),
               (0.237619, 0.061825), (0.295455, 0.212095), (-0.193158, 0.64114), (-0.391662, 0.05246),
               (-0.410059, 0.092665), (-0.096947, 0.41766), (-0.513631, 0.000733), (-0.04113, 0.151642),
               (0.066661, 0.477912), (0.37982, -0.102767), (0.164253, -0.152997), (0.091188, 0.089058),
               (-0.351559, -0.275293), (-0.412911, 0.188302), (-0.129267, -0.382105), (-0.1229, -0.199945),
               (-0.132028, 0.004491), (0.318966, -0.290955), (0.255146, -0.121014), (0.111273, -0.03108),
               (0.485613, -0.124929), (0.130103, -0.225676)]


def test_nmds_matches_metamds(meadow, capsys):
    meadow.add_nmds()
    text = capsys.readouterr().out
    assert "stress = 0.195, usable" in text  # m1$stress = 0.194902988398863
    assert "Clarke" in text and 'metaMDS(x, distance = "bray", k = 2, autotransform = FALSE)' in text
    points = meadow._frame.to_pandas()[["NMDS1", "NMDS2"]].to_numpy()
    _, _, disparity = procrustes(np.array(R_NMDS_BRAY), points)
    assert np.sqrt(1 - disparity) > 0.999  # vegan::procrustes / protest correlation
    assert abs(points.mean(axis=0)).max() < 1e-9
    meadow.undo()
    assert "NMDS1" not in meadow.column_names()


@pytest.mark.parametrize("measure, dimensions, r_stress", [
    ("euclidean", 2, 0.156635223496078),   # metaMDS(sp, distance = "euclidean", k = 2, autotransform = FALSE)
    ("bray-curtis", 3, 0.126865007532683),  # metaMDS(sp, distance = "bray", k = 3, autotransform = FALSE)
    ("jaccard", 2, 0.0103910876886383),   # metaMDS(sp, distance = "jaccard", binary = TRUE, k = 2, ...)
])
def test_nmds_stress_matches_metamds(measure, dimensions, r_stress):
    from simplydata.table.patterns import _distances, _nmds

    x = sdata.example("meadow")[SPECIES].to_numpy(float)
    points, stress, found, starts = _nmds(_distances(x, measure), 30, dimensions, 10, 1)
    assert stress == pytest.approx(r_stress, abs=0.005)
    assert points.shape == (30, dimensions) and starts == 11 and found >= 2


def test_nmds_empty_cells_and_errors(engine, capsys):
    data = sdata.example("meadow")
    data.loc[0, "clover"] = None
    t = make(data, engine)
    t.add_nmds(tries=3)
    assert "1 row with an empty cell" in capsys.readouterr().out
    frame = t._frame.to_pandas()
    assert frame.loc[0, ["NMDS1", "NMDS2"]].isna().all() and frame["NMDS1"][1:].notna().all()
    with pytest.raises(SimplydataError, match="dimensions="):
        t.add_nmds(dimensions=0)
    with pytest.raises(SimplydataError, match="tries="):
        t.add_nmds(tries=-1)


def test_pava_matches_scipy():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.normal(size=rng.integers(1, 60)).cumsum() * rng.choice([-1, 1])
        assert np.allclose(_pava(y), _isotonic(y))
    assert _pava([3, 1, 2]).tolist() == [2, 2, 2]


# ---------------------------------------------------------------------------- PERMANOVA
def test_compare_profiles_matches_adonis2(meadow, capsys):
    result = row(meadow.compare_profiles(for_each="grazed"))
    text = capsys.readouterr().out
    # R: set.seed(1); adonis2(vegdist(sp) ~ grazed, data = meadow, permutations = 999)
    assert result["statistic"] == pytest.approx(9.1438238530611, rel=1e-9)
    assert result["R squared"] == pytest.approx(0.246173465856223, rel=1e-9)
    assert result["p-value"] == pytest.approx(0.001)  # R: 0.001
    assert (result["degrees of freedom 1"], result["degrees of freedom 2"]) == (1, 28)
    # R: b <- betadisper(vegdist(sp), meadow$grazed); b$group.distances; anova(b); permutest(b): p = 0.117
    assert result["distance to centre in no"] == pytest.approx(0.240009218829533, rel=1e-4)
    assert result["distance to centre in yes"] == pytest.approx(0.278142465135866, rel=1e-4)
    assert result["spread check F"] == pytest.approx(2.53042696894578, rel=1e-3)
    assert result["spread check p-value"] == pytest.approx(0.117, abs=0.03)
    assert result["count in no"] == 15 and result["permutations"] == 999
    assert "explain 24.6 % of the differences" in text and "pseudo-F = 9.14" in text
    assert "adonis2" in text and "betadisper" in text and "differ in spread" not in text
    assert meadow._history == []


@pytest.mark.parametrize("for_each, measure, f, r2, r_p, disp_f", [
    # adonis2(vegdist(sp, m) ~ field); p from 9999 permutations in R (0.047); anova(betadisper(...))$F
    ("field", "bray-curtis", 1.80523021588802, 0.1179485829631, 0.047, 0.624609336020348),
    ("field", "euclidean", 1.8941032091796, 0.123040828260144, 0.046, 0.518698457904506),
    ("grazed", "jaccard", 1.7290369552548, 0.0581598710330635, 0.244, None),
])
def test_compare_profiles_other_cases(meadow, capsys, for_each, measure, f, r2, r_p, disp_f):
    result = row(meadow.compare_profiles(for_each=for_each, measure=measure))
    assert result["statistic"] == pytest.approx(f, rel=1e-9)
    assert result["R squared"] == pytest.approx(r2, rel=1e-9)
    assert result["p-value"] == pytest.approx(r_p, abs=0.03)
    if disp_f is not None:
        assert result["spread check F"] == pytest.approx(disp_f, rel=1e-3)
    if measure == "jaccard":  # grazed plots hold more kinds of plant: the groups differ in spread
        assert result["spread check p-value"] < 0.05
        assert "PERMANOVA can then mistake a difference in spread" in capsys.readouterr().out


def test_compare_profiles_skips_empty_cells(engine, capsys):
    data = sdata.example("meadow")
    data.loc[3, "moss"] = None
    data.loc[5, "grazed"] = None
    full = row(make(data, engine).compare_profiles(for_each="grazed", permutations=99))
    assert "2 rows with an empty cell" in capsys.readouterr().out
    fewer = row(make(data.drop(index=[3, 5]), engine).compare_profiles(for_each="grazed", permutations=99))
    assert full["statistic"] == pytest.approx(fewer["statistic"], rel=1e-12)
    assert full["count in no"] + full["count in yes"] == 28


def test_compare_profiles_errors(meadow):
    with pytest.raises(SimplydataError, match="for_each="):
        meadow.compare_profiles()
    with pytest.raises(SimplydataError, match="not found"):
        meadow.compare_profiles(for_each="grazing")
    with pytest.raises(SimplydataError, match="permutations="):
        meadow.compare_profiles(for_each="grazed", permutations=0)
    with pytest.raises(SimplydataError, match="at least 2 rows in each group"):
        meadow.compare_profiles(for_each="plot")
