"""Tests for distributions: sdata.distribution(...), its words, fit_distribution,
compare_distributions and the shape_test engine of compare_shape_to.

Every printed number is checked against R 4.6.2 (values hard-coded below, made by the scripts
named next to them; extraDistr 1.10 and statmod for the families base R lacks, fitdistrplus,
goftest and numDeriv for the fits). Samples made in R with set.seed(42) are written into this
file, so the R fits can be repeated exactly. Fits are compared with R's EXACT maximum (score
equations solved with uniroot(), or optim() with reltol = 1e-15 then BFGS), and standard errors
with numDeriv::hessian() at that maximum, because R's MASS::fitdistr() defaults stop a little
short on some data (noted where it happens).
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.distributions import Distribution, distribution, fit, shape_test
from simplydata.numbers.numbers import Numbers
from simplydata.table import table as make_table

ENGINES = ["pandas", "polars"]
EXACT = dict(rel=1e-10, abs=1e-14)  # d/p/q: R and SciPy use the same closed forms


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        data = pl.from_pandas(data)
    return make_table(data)


def nums(values):
    return sdata.numbers(values)


def row(result):
    frame = result._frame.to_pandas()
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


@pytest.fixture
def days():
    return sdata.example("trial")["days"].to_numpy(float)


# Samples made in R 4.6.2 by ref_fit2.R after set.seed(42), rounded to 4 decimals (R: round(r...(n, ...), 4)).
COUNTS = [
    7, 7, 7, 7, 9, 6, 16, 10, 12, 5, 8, 10, 17, 5, 4, 0, 6, 9, 2, 1, 0, 3, 2, 2, 0, 2, 3, 3, 4, 0, 6, 0, 2,
    4, 2, 6, 7, 2, 4, 1, 11, 1, 8, 2, 1, 5, 0, 1, 5, 6, 8, 4, 5, 0, 0, 9, 3, 1, 2, 4, 4, 6, 3, 5, 0, 11, 1,
    0, 1, 0, 7, 5, 1, 3, 11, 2, 2, 8, 0, 3,
]
T_VALUES = [
    1.6506, 3.3659, 0.6794, 2.3249, 8.5622, 2.8589, 2.6044, 0.615, 5.1281, 0.6894, 2.9862, 1.8276, 3.768,
    0.8087, -2.5338, 2.7726, -1.9117, 3.2058, 6.9826, -1.0668, 4.4695, -2.9976, 5.3358, 2.1599, 8.0427,
    4.1274, 2.8097, 1.947, 5.2801, 2.8531, -0.9781, 2.2598, 4.4459, 3.9167, 3.8245, 9.6096, 2.8997, 5.6928,
    1.4271, 8.0366, -0.4181, 1.6762, 4.444, 4.0154, 0.0678, 1.9327, 1.3202, 0.842, 5.2049, 1.9535, 1.2812,
    3.9393, 9.2106, -0.7613, 1.7871, 2.6535, 2.4435, 0.6761, -0.0337, 2.5604,
]
CAUCHY_VALUES = [
    2.6153, 2.9026, -8.6722, -1.5359, 1.7066, -32.3331, -0.0018, -1.5023, 4.0236, 1.7616, -7.3301, -0.1712,
    -30.2567, -1.8646, 6631.7298, 5.0554, 0.778, 0.5016, 2.4976, 0.7147, 0.1641, -1.8016, -49.5239, 34.1456,
    2.4704, 1.6514, 24.1674, 3.9867, -0.7366, 0.3576, 4.5122, -0.2347, 2.6549, -0.807, -9.3592, 9.1561,
    -0.1112, 3.534, 22.6816, -0.6456, 1.5317, 6.8128, 1.6198, 1.8905, 2.6665, 3.3286, 3.7357, 5.8193,
    75.4623, 0.198,
]
LOGISTIC_VALUES = [
    3.1837, 7.1188, -0.399, 4.0896, -2.7552, -2.8974, 2.9701, 1.3886, 1.8822, -2.9296, 4.9182, 4.3516,
    -5.0532, 2.7619, 1.2788, 6.7054, 5.6774, 0.6469, 3.3664, 1.0404, -1.7409, 2.8606, 3.395, 2.7557, -1.879,
    -0.7146, -0.4606, 1.579, 5.0325, 4.9306, 0.0704, 3.0688, 3.7269, 2.9304, 0.0669, -0.7122, -4.5946,
    2.4499, -1.7978, 2.0226, -2.6609, -0.0381, -0.5621, 3.9681, -0.0789, 0.5148, 0.51, 1.8262, 0.2216,
    7.2702,
]
BETA_VALUES = [
    0.0826, 0.2092, 0.3615, 0.3039, 0.5976, 0.0605, 0.6571, 0.166, 0.2331, 0.1115, 0.3135, 0.1153, 0.3941,
    0.6405, 0.2478, 0.2437, 0.2281, 0.4219, 0.3548, 0.2286, 0.3971, 0.1873, 0.2326, 0.5568, 0.3834, 0.3369,
    0.2369, 0.5104, 0.1411, 0.2971, 0.2957, 0.1828, 0.1246, 0.2852, 0.4366, 0.6658, 0.0426, 0.1642, 0.2269,
    0.1573,
]
GUMBEL_VALUES = [
    8.9373, 9.2038, 5.5486, 14.6594, 10.871, 13.1469, 27.9872, 11.4639, 7.5698, 16.0298, 17.6385, 8.8083,
    10.6751, 9.7126, 11.4071, 8.3064, 19.4872, 11.9089, 10.494, 10.9908, 16.3286, 13.5994, 15.8065, 10.0953,
    19.4872, 10.2254, 6.0814, 11.2947, 9.217, 13.2003, 7.3615, 15.6978, 12.4809, 11.5888, 16.5146, 7.0857,
    11.2547, 11.4744, 10.6313, 10.1531, 9.3975, 19.0648, 12.6383, 12.8558, 7.113, 6.751, 9.2201, 5.3987,
    15.9901, 15.3252,
]
LAPLACE_VALUES = [
    0.8333, -1.3856, 2.057, 1.1851, 0.8261, -1.3189, 3.0599, 0.8722, -1.0544, -0.1683, 0.9445, 3.1474,
    -1.4672, 1.8738, -3.4015, -1.6903, 0.0497, 1.4777, 2.2096, 0.2822, 0.347, -0.2637, 1.0982, -3.0227,
    -1.1854, -2.6952, 3.3949, -1.775, -6.5611, 1.1746, -0.0021, 2.2623, -3.3494, 2.3966, 0.8426, 8.7766,
    0.1204, -0.2475, 0.7404, -2.8893, -0.3404,
]
PARETO_VALUES = [
    3.1018, 2.1558, 2.9912, 2.097, 5.8764, 2.866, 2.9564, 3.9536, 2.7763, 2.5816, 5.5575, 2.0084, 2.065,
    2.0139, 3.607, 5.7205, 3.8261, 2.8099, 2.4494, 2.0436, 2.3262, 2.546, 4.5329, 2.0381, 2.8469, 2.7875,
    2.3403, 2.2238, 3.4318, 2.3863, 2.1172, 2.0004, 8.8765, 2.1356, 3.3885, 4.1271, 2.2654, 3.0832, 2.134,
    2.3155, 2.4746, 6.5361, 3.0314, 6.0806, 2.0731, 3.9383, 5.3776, 2.8012, 2.2574, 2.0774,
]
HALFNORMAL_VALUES = [
    1.3852, 0.6118, 3.4693, 3.2744, 0.925, 3.3786, 2.5591, 3.7601, 1.1949, 3.9263, 0.1681, 3.3854, 3.6811,
    0.1311, 0.0247, 2.7206, 0.3223, 0.8949, 2.9097, 0.7252, 0.5495, 2.2852, 0.729, 1.316, 2.4723, 1.6343,
    2.6525, 2.5943, 0.7785, 1.6151, 1.5357, 0.2104, 1.286, 5.0596, 0.4869, 3.1924, 0.1517, 3.1801, 1.0549,
    1.3999, 0.8866, 2.4664, 4.7263, 1.7167, 3.0153, 0.055, 1.9836, 0.3346, 0.2613, 1.391,
]
INVGAUSS_VALUES = [
    2.6592, 2.3334, 1.3465, 0.8293, 0.3918, 0.4705, 0.3235, 0.8012, 3.5393, 1.0583, 2.1308, 0.7258, 1.3347,
    5.5756, 1.396, 0.7721, 1.7933, 1.7399, 0.9695, 5.888, 2.0125, 1.4077, 0.431, 1.5675, 1.8129, 1.6348,
    4.5392, 2.7496, 1.3457, 0.9378, 0.8446, 2.3437, 0.4775, 3.5043, 2.0995, 1.9511, 0.7341, 1.6354, 7.1076,
    4.2701, 3.9094, 3.4863, 1.1557, 0.5957, 0.5356, 1.3064, 0.6597, 1.2554, 1.0985, 1.728,
]
CHISQ_VALUES = [
    3.5229, 3.1053, 2.6686, 3.1325, 5.0485, 8.0424, 3.6859, 6.2171, 3.5383, 0.4397, 3.0925, 1.6748, 2.5892,
    1.9433, 1.1751, 1.7898, 0.7402, 1.2766, 0.6286, 2.6167, 6.6593, 1.9492, 2.5532, 2.187, 10.4382, 6.4257,
    1.2586, 1.3502, 2.8298, 9.4091, 2.8311, 1.6454, 1.8163, 4.7413, 2.3696, 2.7376, 2.8587, 1.1846, 1.9252,
    5.5052, 4.2191, 8.7129, 1.9802, 7.6675, 2.1913, 6.5867, 6.3246, 3.4394, 2.3659, 1.5358,
]
F_VALUES = [
    0.4868, 1.5133, 1.4512, 0.8078, 0.3915, 0.5268, 1.5115, 0.5569, 0.9974, 1.5855, 0.8931, 1.7309, 0.7876,
    1.0978, 1.6886, 0.6033, 0.2171, 0.2578, 0.9635, 1.8259, 0.3229, 5.4299, 0.3202, 0.427, 2.0324, 0.4562,
    1.7033, 1.1859, 1.1748, 0.216, 3.8611, 0.7115, 0.0537, 0.2641, 0.2534, 0.342, 1.5393, 0.1687, 1.468,
    0.4491, 0.3185, 1.6265, 1.0074, 0.4438, 1.8323, 1.1348, 1.4906, 1.485, 0.5869, 3.5174, 1.5712, 1.9619,
    2.1252, 0.4874, 0.2655, 0.6911, 0.5231, 0.6434, 2.263, 1.8678, 0.8, 0.5193, 1.4446, 0.6146, 0.7691,
    1.6545, 0.4519, 0.669, 1.1496, 1.4907, 1.2598, 4.5459, 0.5023, 2.2079, 1.096, 1.7294, 1.3855, 0.2198,
    3.0539, 1.2126,
]
ANGLE_VALUES = [
    0.138, 0.2777, 1.1359, 1.1331, 0.8498, -0.0919, 1.5108, -0.3201, 0.2401, 1.895, 0.8145, 1.0824, 0.7262,
    -0.0976, 0.1415, 0.5992, -1.2571, -0.0087, 0.9792, 0.3209, 0.3298, 1.0217, 0.1734, 0.8773, -0.3534,
    -0.2365, -0.5045, 0.5506, 0.3763, 1.3651, 0.4749, 1.3123, 1.6671, 0.2054, 0.7331, -0.0069, 0.9428,
    -0.1479, -0.1159, 0.6733, 0.5545, 0.6576, 0.5416, 0.1828, 0.4219, 1.4721, 0.4893, -0.2911, -0.0067,
    -0.1611,
]


# ====================================================================== the catalogue against R
# (name, parameters, x, R density/chance at x, R p(x), R p(x, lower.tail = FALSE), R q(0.3), R q(0.95))
# Made by ref_catalogue.R, e.g. dgamma(7, 2.5, 0.3), pgamma(7, 2.5, 0.3), qgamma(.3, 2.5, .3);
# extraDistr::dlaplace/dgumbel/dpareto/dtriang/dhnorm/ddunif, statmod::dinvgauss; von Mises by
# integrate(rel.tol = 1e-13) and uniroot(tol = 1e-14) on its density.
CATALOGUE = [
    ("normal", dict(average=50, standard_deviation=10), 60, 0.024197072451914336, 0.84134474606854293,
     0.15865525393145705, 44.755994872919594, 66.448536269514719),
    ("lognormal", dict(log_average=1, log_standard_deviation=0.5), 3, 0.26083887270192596, 0.57817410080287324,
     0.42182589919712676, 2.0913289886882174, 6.18685458819545),
    ("exponential", dict(rate=0.5), 2, 0.18393972058572117, 0.63212055882855767, 0.36787944117144233,
     0.71334988787746467, 5.9914645471079799),
    ("gamma", dict(shape=2.5, rate=0.3), 7, 0.084099920466536859, 0.47900504656859499, 0.52099495343140501,
     4.9998468879331783, 18.450829489193922),
    ("beta", dict(shape1=2, shape2=5), 0.3, 2.1608999999999998, 0.57982500000000015, 0.42017499999999985,
     0.18180347131894911, 0.58180340925202589),
    ("uniform", dict(between=[2, 7]), 3.5, 0.2, 0.3, 0.7, 3.5, 6.75),
    ("t", dict(degrees_of_freedom=5), 1.2, 0.1776586134649355, 0.85805447164694892, 0.14194552835305113,
     -0.55942964446936105, 2.0150483733330224),
    ("t", dict(degrees_of_freedom=4, location=2, scale=3), 4, 0.096054183927614514, 0.72926513036220753,
     0.27073486963779247, 0.29405281085088353, 8.3955403589799467),
    ("chi-square", dict(degrees_of_freedom=4), 3, 0.1673476201113224, 0.44217459962892536, 0.55782540037107464,
     2.1946984214069829, 9.487729036781154),
    ("F", dict(degrees_of_freedom=[3, 20]), 1.5, 0.25517341781194397, 0.75494798840603283, 0.24505201159396714,
     0.47974723393079177, 3.0983912121407795),
    ("weibull", dict(shape=1.5, scale=3), 2, 0.23687782228285617, 0.41977020402532722, 0.58022979597467284,
     1.5088161447471551, 6.2343319126036691),
    ("cauchy", dict(location=1, scale=2), 3, 0.079577471545947673, 0.75, 0.25, -0.45308505601072202,
     13.627503029350075),
    ("logistic", dict(location=1, scale=2), 3, 0.098305966620740939, 0.7310585786300049, 0.2689414213699951,
     -0.69459572077440712, 6.8888779583328787),
    ("laplace", dict(location=1, scale=2), 3, 0.09196986029286057, 0.81606027941427883, 0.18393972058572117,
     -0.021651247531981443, 5.60517018598809),
    ("gumbel", dict(location=1, scale=2), 3, 0.12732319002179124, 0.69220062755534639, 0.30779937244465361,
     0.62874648227526853, 6.9403904980843274),
    ("pareto", dict(shape=3, scale=2), 3, 0.29629629629629611, 0.70370370370370372, 0.29629629629629628,
     2.2524957608872125, 5.4288352331898118),
    ("triangular", dict(between=[0, 10], peak=3), 4, 0.1714285714285714, 0.48571428571428588,
     0.51428571428571412, 3, 8.1291713066130278),
    ("half-normal", dict(standard_deviation=2), 1.5, 0.30113743215480443, 0.54674529524626347,
     0.45325470475373653, 0.77064093281513524, 3.9199279690801072),
    ("inverse gaussian", dict(average=2, shape=3), 1.5, 0.35333804312537148, 0.4956901248416295,
     0.50430987515837045, 1.0266971096801285, 5.1773955292983143),
    ("von mises", dict(average=0.5, concentration=2), 1, 0.40385253335183785, 0.73819221441852623,
     0.26180778558147377, 0.090740763537029007, 1.9179661935321275),
    ("binomial", dict(tries=10, chance=0.3), 3, 0.26682793199999993, 0.64961071840000018, 0.35038928159999977,
     2, 5),
    ("bernoulli", dict(chance=0.3), 1, 0.3, 1, 0, 0, 1),
    ("poisson", dict(average=3.5), 2, 0.18495897346170084, 0.32084719886213409, 0.67915280113786591, 2, 7),
    ("negative binomial", dict(size=3, chance=0.4), 4, 0.124416, 0.58009599999999995, 0.419904, 2, 11),
    ("negative binomial", dict(size=3, average=4), 4, 0.12589506558856065, 0.64065507204845429,
     0.35934492795154571, 2, 10),
    ("geometric", dict(chance=0.2), 3, 0.1024, 0.59040000000000004, 0.40959999999999996, 1, 13),
    ("hypergeometric", dict(marked=20, unmarked=30, drawn=10), 4, 0.28005860310537128, 0.64502688988220791,
     0.35497311011779209, 3, 6),
    ("whole number uniform", dict(between=[1, 6]), 3, 1 / 6, 0.5, 0.5, 2, 6),
]


@pytest.mark.parametrize("name, params, x, d, p, upper, q30, q95", CATALOGUE,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(CATALOGUE)])
def test_every_catalogue_distribution_matches_r(name, params, x, d, p, upper, q30, q95, capsys):
    dist = distribution(name, **params)
    tolerance = dict(rel=1e-8, abs=1e-12) if name == "von mises" else EXACT  # R side: integrate()/uniroot()
    height = dist.chance(x) if dist.is_discrete else dist.density_at(x)
    assert height == pytest.approx(d, **tolerance)
    assert dist.chance(at_most=x) == pytest.approx(p, **tolerance)
    assert dist.chance(above=x) == pytest.approx(upper, **tolerance)
    assert dist.percentile(30) == pytest.approx(q30, **tolerance)
    assert dist.percentile(95) == pytest.approx(q95, **tolerance)
    out = capsys.readouterr().out
    if name != "von mises":
        assert "R: " in out


def test_printed_sentence_names_the_r_call(capsys):
    dist = sdata_distribution = distribution("normal", average=50, standard_deviation=10)
    assert repr(sdata_distribution) == ("Normal distribution with average 50 and standard deviation 10 "
                                        "(R: dnorm(x, mean = 50, sd = 10)).")
    share = dist.chance(above=80)
    assert share == pytest.approx(0.0013498980316300946, **EXACT)  # pnorm(80, 50, 10, lower.tail = FALSE)
    assert capsys.readouterr().out.strip() == (
        "The chance of a value above 80 is 0.135 % (normal, average 50, standard deviation 10; "
        "R: pnorm(80, mean = 50, sd = 10, lower.tail = FALSE)).")


def test_tiny_upper_tails_stay_accurate(capsys):
    dist = distribution("normal", average=50, standard_deviation=10)
    assert dist.chance(above=130) == pytest.approx(6.2209605742717849e-16, rel=1e-10)  # 1 - pnorm would give 0
    assert distribution("normal").cut_off(above=1e-12) == pytest.approx(7.0344838253011321, rel=1e-10)
    assert dist.cut_off(above=0.025) == pytest.approx(69.599639845400532, **EXACT)
    assert dist.chance(between=[40, 60]) == pytest.approx(0.68268949213708585, **EXACT)
    assert distribution("t", df=7).cut_off(above=0.025) == pytest.approx(2.3646242515927836, **EXACT)
    out = capsys.readouterr().out
    assert "R: pnorm(60, mean = 50, sd = 10) - pnorm(40, mean = 50, sd = 10)" in out
    assert "R: qt(0.025, df = 7, lower.tail = FALSE)" in out


def test_counts_spell_out_strict_and_inclusive_limits(capsys):
    dist = distribution("binomial", tries=10, chance=0.3)
    assert dist.chance(below=3) == pytest.approx(0.38278278639999974, **EXACT)  # pbinom(2, 10, .3)
    assert "fewer than 3 (2 or fewer)" in capsys.readouterr().out
    assert dist.chance(at_most=3) == pytest.approx(0.64961071840000018, **EXACT)  # pbinom(3, 10, .3)
    assert "3 or fewer" in capsys.readouterr().out
    assert dist.chance(above=3) == pytest.approx(0.35038928159999977, **EXACT)
    assert "more than 3 (4 or more)" in capsys.readouterr().out
    assert dist.chance(at_least=3) == pytest.approx(0.61721721360000026, **EXACT)  # pbinom(2, lower.tail = F)
    assert "R: pbinom(2, size = 10, prob = 0.3, lower.tail = FALSE)" in capsys.readouterr().out
    assert dist.chance(between=[2, 5]) == pytest.approx(0.8033426667000001, **EXACT)  # pbinom(5) - pbinom(1)
    assert "2 to 5 (both included)" in capsys.readouterr().out
    assert dist.chance(above=1, below=6) == pytest.approx(0.8033426667000001, **EXACT)
    assert dist.chance(3) == pytest.approx(0.26682793199999993, **EXACT)
    assert dist.chance(below=2.5) == pytest.approx(dist.chance(at_most=2))
    assert dist.chance(above=7, below=7) == 0


def test_counts_cut_off_and_middle_range_match_r(capsys):
    dist = distribution("binomial", size=10, prob=0.3)  # R's names work too
    assert dist.cut_off(above=0.05) == 5  # qbinom(.05, 10, .3, lower.tail = FALSE)
    assert "4.73 %" in capsys.readouterr().out  # pbinom(5, 10, .3, lower.tail = FALSE) = 0.0473
    middle = dist.middle_range(95)
    assert list(middle._values) == [0, 6]  # qbinom(c(.025, .975), 10, .3)
    normal = distribution("normal", mean=50, sd=10).middle_range(95)
    assert list(normal._values) == pytest.approx([30.400360154599468, 69.599639845400532], **EXACT)


def test_lists_give_numbers(capsys):
    dist = distribution("normal", average=50, standard_deviation=10)
    got = dist.chance(below=[50, 60, 70])
    assert isinstance(got, Numbers)
    assert list(got._values) == pytest.approx(stats.norm(50, 10).cdf([50, 60, 70]), **EXACT)
    assert "The chances of a value below 50, 60 and 70 are 50 %, 84.1 % and 97.7 %" in capsys.readouterr().out
    assert isinstance(dist.percentile([5, 50, 95]), Numbers)
    assert isinstance(dist.density_at([40, 50]), Numbers)
    assert isinstance(dist.cut_off(above=[0.05, 0.025]), Numbers)
    assert isinstance(distribution("poisson", average=2).chance([0, 1, 2]), Numbers)
    assert isinstance(dist.chance(below=sdata.numbers([1, 2])), Numbers)
    assert isinstance(dist.chance(above=60), float)


def test_moments_and_summary_match_formulas(capsys):
    dist = distribution("gamma", shape=2, scale=10)  # scale= in place of rate=
    assert dist.parameters() == {"shape": 2.0, "rate": 0.1}
    assert dist.average() == pytest.approx(20)
    assert dist.variance() == pytest.approx(200)
    assert dist.standard_deviation() == pytest.approx(math.sqrt(200))
    assert dist.median() == pytest.approx(stats.gamma(2, scale=10).median())
    got = row(dist.summary())
    assert got["skewness"] == pytest.approx(2 / math.sqrt(2))
    assert got["excess kurtosis"] == pytest.approx(6 / 2)
    assert got["lowest possible"] == 0 and math.isinf(got["highest possible"])
    assert distribution("cauchy").average() != distribution("cauchy").average()  # nan: no average
    assert "no finite average" in capsys.readouterr().out


def test_scipy_distributions_by_their_own_names(capsys):
    dist = distribution("burr12", c=2, d=3)
    assert repr(dist) == "SciPy's burr12 with c=2, d=3."
    assert dist.chance(below=1) == pytest.approx(stats.burr12(2, 3).cdf(1), **EXACT)
    with pytest.raises(SimplydataError, match="needs d="):
        distribution("burr12", c=2)


def test_random_numbers_repeat_with_the_seed(capsys):
    dist = distribution("normal", average=50, standard_deviation=10)
    first = dist.random_numbers(how_many=2000, seed=3)
    again = dist.random_numbers(how_many=2000, seed=3)
    assert np.array_equal(first._values, again._values)
    assert first._values.mean() == pytest.approx(50, abs=1)  # 4 standard errors
    assert first._values.std() == pytest.approx(10, abs=0.7)
    assert "seed=3; R: rnorm(2000, mean = 50, sd = 10), whose numbers differ" in capsys.readouterr().out
    grid = distribution("poisson", average=4).random_numbers(rows=3, columns=2)
    assert grid._values.shape == (3, 2)
    assert np.all(grid._values == np.round(grid._values))


def test_friendly_errors():
    with pytest.raises(SimplydataError, match='Did you mean "normal"'):
        distribution("normall")
    with pytest.raises(SimplydataError, match="needs shape="):
        distribution("gamma", rate=2)
    with pytest.raises(SimplydataError, match="Did you mean standard_deviation="):
        distribution("normal", standard_deviaton=2)
    with pytest.raises(SimplydataError, match="a number above 0"):
        distribution("normal", sd=-1)
    with pytest.raises(SimplydataError, match="chance from 0 to 1"):
        distribution("binomial", tries=10, chance=1.5)
    with pytest.raises(SimplydataError, match="with their names"):
        distribution("poisson", 3)
    with pytest.raises(SimplydataError, match="not both"):
        distribution("gamma", shape=2, rate=1, scale=1)
    with pytest.raises(SimplydataError, match="not both and not neither"):
        distribution("negative binomial", size=2, chance=0.3, average=4)
    with pytest.raises(SimplydataError, match="peak"):
        distribution("triangular", between=[0, 10], peak=12)
    with pytest.raises(SimplydataError, match="lowest value"):
        distribution("uniform", between=[5, 1])
    normal, counts = distribution("normal"), distribution("poisson", average=2)
    with pytest.raises(SimplydataError, match="density_at"):
        normal.chance(3)
    with pytest.raises(SimplydataError, match="chance\\(3\\)"):
        counts.density_at(3)
    with pytest.raises(SimplydataError, match="0 to 100"):
        normal.percentile(120)
    with pytest.raises(SimplydataError, match="percentile\\(97.5\\)"):
        normal.cut_off(above=2.5)
    with pytest.raises(SimplydataError, match="a range"):
        normal.chance()
    with pytest.raises(SimplydataError, match="no option called"):
        normal.chance(over=3)
    with pytest.raises(SimplydataError, match="density_at"):
        normal.pdff  # noqa: B018
    with pytest.raises(SimplydataError, match="dist.percentile"):
        normal.ppf  # noqa: B018
    with pytest.raises(SimplydataError, match="cannot be changed"):
        normal.average_value = 3


def test_interface_for_charts():
    dist = distribution("gamma", shape=2, rate=0.5)
    assert dist.is_discrete is False and dist.label == "gamma, shape 2, rate 0.5"
    assert dist.r_call == "dgamma(x, shape = 2, rate = 0.5)"
    assert list(dist.pdf([1, 2])) == pytest.approx(stats.gamma(2, scale=2).pdf([1, 2]))
    counts = distribution("poisson", average=2)
    assert counts.is_discrete and list(counts.pmf([0, 1])) == pytest.approx(stats.poisson(2).pmf([0, 1]))


def test_chart_draws_with_the_curve_word(monkeypatch):
    import sys

    module = sys.modules["simplydata.chart"]
    Chart = module.Chart
    if not hasattr(module.Chart, "curve"):  # until the chart words for curves are added to Chart
        curves = pytest.importorskip("simplydata.words.curves")
        monkeypatch.setattr(module, "Chart", type("Chart", (curves.CurveWords, module.Chart), {}))
        Chart = module.Chart
    for dist in (distribution("normal", average=50, standard_deviation=10), distribution("poisson", average=3)):
        for show in ("density", "cumulative"):
            assert isinstance(dist.chart(show=show), Chart)
    with pytest.raises(SimplydataError, match="cumulative"):
        distribution("normal").chart(show="cdf")


# ====================================================================== fitting against R
# ref_fit.R / ref_fit2.R. Gamma: exact maximum from log(a) - digamma(a) = log(mean(d)) - mean(log(d)),
# uniroot(tol = 1e-15); fitdistr(d, "gamma") with its defaults stops short (shape 2.1280434,
# log-likelihood -843.8147634). Weibull: its score equation with uniroot(); fitdistr(d, "weibull")
# also stops short (shape 1.69071, log-likelihood -840.18972). Standard errors: numDeriv::hessian().
GAMMA = dict(shape=2.12734096971, rate=0.00446623248479, se=(0.2560323444, 0.0006058646734),
             loglik=-843.814759302, aic=1691.6295186, bic=1697.20450209)


def test_fit_gamma_matches_the_exact_r_maximum(days, engine, capsys):
    t = make(sdata.example("trial"), engine)
    fitted = t.fit_distribution("days", "gamma")
    assert isinstance(fitted, Distribution)
    got = row(fitted.table())
    assert got["shape"] == pytest.approx(GAMMA["shape"], rel=1e-10)
    assert got["rate"] == pytest.approx(GAMMA["rate"], rel=1e-10)
    assert got["standard error of shape"] == pytest.approx(GAMMA["se"][0], rel=1e-6)
    assert got["standard error of rate"] == pytest.approx(GAMMA["se"][1], rel=1e-6)
    assert got["log-likelihood"] == pytest.approx(GAMMA["loglik"], rel=1e-11)
    assert got["AIC"] == pytest.approx(GAMMA["aic"], rel=1e-11)
    assert got["BIC"] == pytest.approx(GAMMA["bic"], rel=1e-11)
    assert got["rows"] == 120 and got["column"] == "days"
    out = capsys.readouterr().out
    assert ('Fitted a gamma distribution to "days" (120 values): shape = 2.13 (standard error 0.256), '
            "rate = 0.00447 (0.000606). Log-likelihood -843.8, AIC 1691.6.") in out
    assert 'as R MASS::fitdistr(x, "gamma")' in out and 'compare_shape_to("days", "gamma")' in out
    assert fitted.chance(above=365) == pytest.approx(stats.gamma(GAMMA["shape"], scale=1 / GAMMA["rate"]).sf(365))
    if engine == "polars":
        assert fitted.table()._engine == "Polars"


def test_fit_closed_forms_match_fitdistr(days, capsys):
    t = make(sdata.example("trial"), "pandas")
    normal = row(t.fit_distribution("days").table())  # fitdistr(d, "normal")
    assert normal["average"] == pytest.approx(476.316666667, rel=1e-11)
    assert normal["standard deviation"] == pytest.approx(281.234243509, rel=1e-11)
    assert normal["standard error of average"] == pytest.approx(25.6730565188, rel=1e-10)
    assert normal["standard error of standard deviation"] == pytest.approx(18.1535923582, rel=1e-10)
    assert normal["log-likelihood"] == pytest.approx(-846.975175449, rel=1e-11)
    assert "a little smaller than R sd()" in capsys.readouterr().out
    lognormal = row(t.fit_distribution("days", "lognormal").table())
    assert lognormal["log average"] == pytest.approx(5.91300526464, rel=1e-11)
    assert lognormal["log standard deviation"] == pytest.approx(0.831056685233, rel=1e-11)
    assert lognormal["standard error of log average"] == pytest.approx(0.0758647488446, rel=1e-10)
    assert lognormal["log-likelihood"] == pytest.approx(-857.62638296, rel=1e-11)
    exponential = row(t.fit_distribution("days", "exponential").table())
    assert exponential["rate"] == pytest.approx(0.00209944364743, rel=1e-11)
    assert exponential["standard error of rate"] == pytest.approx(0.000191652107318, rel=1e-10)
    assert exponential["log-likelihood"] == pytest.approx(-859.929947898, rel=1e-11)
    weibull = row(t.fit_distribution("days", "weibull").table())
    assert weibull["shape"] == pytest.approx(1.68720733202, rel=1e-10)
    assert weibull["scale"] == pytest.approx(531.396176566, rel=1e-10)
    assert weibull["standard error of shape"] == pytest.approx(0.1254202971, rel=1e-6)
    assert weibull["standard error of scale"] == pytest.approx(30.15506407, rel=1e-6)
    assert weibull["log-likelihood"] == pytest.approx(-840.18570827, rel=1e-11)


def test_fit_with_a_fixed_parameter(capsys):
    t = make(sdata.example("trial"), "pandas")
    got = row(t.fit_distribution("days", "gamma", fixed={"shape": 2}).table())
    # exact: rate = 2 / mean(d); fitdistrplus::fitdist(fix.arg = list(shape = 2)) from rate 0.004 stops at
    # 0.0042649 (log-likelihood -843.9733), short of the maximum -843.943940704.
    assert got["rate"] == pytest.approx(2 / 476.316666667, rel=1e-8)
    assert got["standard error of rate"] == pytest.approx(0.0002710370094, rel=1e-6)
    assert got["log-likelihood"] == pytest.approx(-843.943940704, rel=1e-10)
    assert got["AIC"] == pytest.approx(2 * 843.943940704 + 2, rel=1e-10)  # one value fitted
    assert math.isnan(got["standard error of shape"]) and got["held fixed"] == "shape"
    assert "shape = 2 (held fixed)" in capsys.readouterr().out


def test_fit_counts_match_fitdistr(capsys):
    counts = nums(COUNTS)
    poisson = row(counts.fit_distribution("poisson").table())  # fitdistr(counts, "poisson")
    assert poisson["average"] == pytest.approx(4.375) and poisson["log-likelihood"] == pytest.approx(-247.688738575)
    assert poisson["standard error of average"] == pytest.approx(0.233853586673, rel=1e-10)
    geometric = row(counts.fit_distribution("geometric").table())
    assert geometric["chance"] == pytest.approx(0.186046511628, rel=1e-10)
    assert geometric["standard error of chance"] == pytest.approx(0.0187661996462, rel=1e-10)
    assert geometric["log-likelihood"] == pytest.approx(-206.588904893, rel=1e-11)
    nb = row(counts.fit_distribution("negative binomial").table())  # score equation in size, mu = mean
    assert nb["size"] == pytest.approx(1.59241553484, rel=1e-10)  # fitdistr: 1.592415433
    assert nb["average"] == pytest.approx(4.375, rel=1e-12)
    assert nb["standard error of size"] == pytest.approx(0.3780812701, rel=1e-6)
    assert nb["standard error of average"] == pytest.approx(0.4526984136, rel=1e-6)
    assert nb["log-likelihood"] == pytest.approx(-204.575450606, rel=1e-11)
    assert "chance" not in nb


# (name, sample, R values in plain order, R standard errors, R log-likelihood). Exact R maxima:
# optim(reltol = 1e-15) then BFGS on a log scale (t, cauchy, logistic, beta, F), optimize(tol = 1e-14)
# (chi-square), closed forms (laplace, pareto, half-normal, inverse gaussian), uniroot on the score
# equation (von Mises). fitdistr(x, "t") stops short here (df 6.1039, log-likelihood -144.068467937).
NUMERIC_FITS = [
    ("t", "T_VALUES", [6.08686889339, 2.60772667739, 2.25231810447], [6.622209923, 0.3448809409, 0.465459888],
     -144.068464468),
    ("cauchy", "CAUCHY_VALUES", [1.36486779878, 2.2546984395], [0.4658197909, 0.4375677693], -183.613120984),
    ("logistic", "LOGISTIC_VALUES", [1.45428658398, 1.67586770775], [0.4163790984, 0.1952888991], -124.907668109),
    ("beta", "BETA_VALUES", [2.17652990342, 5.1382033819], [0.4565773609, 1.15103843], 19.2444475072),
    ("gumbel", "GUMBEL_VALUES", [10.0603810286, 3.28862995121], [0.4896856093, 0.3638131892], -138.459812939),
    ("laplace", "LAPLACE_VALUES", [0.2822, 1.76018780488], [math.nan, 1.76018780488 / math.sqrt(41)],
     -92.6012753406),
    ("pareto", "PARETO_VALUES", [2.54883089897, 2.0004], [2.54883089897 / math.sqrt(50), math.nan],
     -57.5024557637),
    ("half-normal", "HALFNORMAL_VALUES", [2.24210570157], [0.224210570158], -76.6603411144),
    ("inverse gaussian", "INVGAUSS_VALUES", [1.90432, 2.71256279511], [0.2256498194, 0.542512559], -73.9642012968),
    ("chi-square", "CHISQ_VALUES", [3.71851316798], [0.3363479499], -105.242920193),
    ("F", "F_VALUES", [4.60275434464, 13.7003054647], [0.9735100957, 7.001479791], -86.6985186785),
    ("von mises", "ANGLE_VALUES", [0.463211048678, 3.23705248241], [0.08648214185, 0.5662426921], -47.152990447),
]


@pytest.mark.parametrize("name, sample, values, errors, loglik", NUMERIC_FITS, ids=[f[0] for f in NUMERIC_FITS])
def test_more_fits_match_the_exact_r_maximum(name, sample, values, errors, loglik, capsys):
    fitted = fit(globals()[sample], name)
    got = [v for v in fitted.parameters().values() if v is not None]
    info = fitted._fit_info
    # numerical searches stop on the log-likelihood, which is flat near the top: values to 1e-5,
    # (t's degrees of freedom, the flattest, to 1e-6 relative here), log-likelihood to 1e-10
    assert got == pytest.approx(values, rel=1e-5)
    assert [info["errors"][k] for k in fitted.parameters() if fitted.parameters()[k] is not None] == pytest.approx(
        errors, rel=1e-4, nan_ok=True)
    assert info["log-likelihood"] == pytest.approx(loglik, rel=1e-10)


def test_fit_for_each_group_gives_a_table(engine, capsys):
    t = make(sdata.example("trial"), engine)
    result = t.fit_distribution("days", "normal", for_each="treatment")
    frame = result._frame.to_pandas()
    assert list(frame["treatment"]) == ["new medicine", "usual care"]
    trial = sdata.example("trial")
    for _, r in frame.iterrows():
        values = trial.loc[trial["treatment"] == r["treatment"], "days"].to_numpy(float)
        assert r["average"] == pytest.approx(values.mean())
        assert r["standard deviation"] == pytest.approx(values.std())
    out = capsys.readouterr().out
    assert 'Fitted a normal distribution to "days" for each "treatment":' in out
    assert "new medicine (60 values)" in out


def test_fit_skips_empty_cells_with_a_note(capsys):
    data = pd.DataFrame({"x": [1.2, 2.5, None, 3.1, 0.7, 1.9]})
    fitted = make(data, "pandas").fit_distribution("x", "exponential")
    assert fitted.parameters()["rate"] == pytest.approx(1 / np.mean([1.2, 2.5, 3.1, 0.7, 1.9]))
    assert 'Note: 1 row with an empty cell in "x" was skipped.' in capsys.readouterr().out
    assert nums([1.0, None, 2.0, 4.0]).fit_distribution("gamma").table() is not None
    assert "1 empty value was skipped" in capsys.readouterr().out


def test_fit_errors_say_how_to_fix():
    t = make(pd.DataFrame({"x": [-1.0, 2, 3, 4], "n": [1, 2, 3, 30]}), "pandas")
    with pytest.raises(SimplydataError, match="only values above 0"):
        t.fit_distribution("x", "gamma")
    with pytest.raises(SimplydataError, match='fixed=\\{"tries": 10\\}'):
        t.fit_distribution("n", "binomial")
    with pytest.raises(SimplydataError, match="give them yourself"):
        t.fit_distribution("x", "triangular")
    with pytest.raises(SimplydataError, match='fit "poisson" instead'):
        fit([2, 3, 2, 3, 2, 3], "negative binomial")
    with pytest.raises(SimplydataError, match="not a parameter"):
        t.fit_distribution("n", "gamma", fixed={"shap": 2})
    binomial = row(t.fit_distribution("n", "binomial", fixed={"tries": 40}).table())
    assert binomial["chance"] == pytest.approx(36 / 4 / 40)


# ====================================================================== comparing and testing shapes
def test_compare_distributions_matches_gofstat(engine, capsys):
    # fitdistrplus::gofstat(list(norm, lnorm, gamma, weibull)), gamma and weibull started at the exact maximum
    result = make(sdata.example("trial"), engine).compare_distributions("days")
    frame = result._frame.to_pandas().set_index("distribution")
    assert list(frame.index) == ["weibull", "gamma", "normal", "lognormal"]
    r = {"normal": (1.446807653051138, 0.1029207153292602, 0.2369524419003307, 1697.95035089818, 1703.52533438374),
         "lognormal": (2.963022691460878, 0.0991842338059668, 0.4062804018172326, 1719.25276591987,
                       1724.82774940543),
         "gamma": (1.054833450264894, 0.0753469507655198, 0.1276334996371588, 1691.62951860371, 1697.20450208927),
         "weibull": (0.656369498499288, 0.0569341361924359, 0.0737789151837476, 1684.37141653941,
                     1689.94640002497)}
    for name, (ad, ks, cvm, aic, bic) in r.items():
        got = frame.loc[name]
        assert got["Anderson-Darling statistic"] == pytest.approx(ad, rel=1e-6)
        assert got["Kolmogorov-Smirnov statistic"] == pytest.approx(ks, rel=1e-6)
        assert got["Cramer-von Mises statistic"] == pytest.approx(cvm, rel=1e-6)
        assert got["AIC"] == pytest.approx(aic, rel=1e-9)
        assert got["BIC"] == pytest.approx(bic, rel=1e-9)
    assert frame.loc["gamma", "AIC difference"] == pytest.approx(1691.62951860371 - 1684.37141653941, rel=1e-8)
    out = capsys.readouterr().out
    assert "Best by AIC: weibull (AIC 1684.4), then gamma (+7.3), normal (+13.6), lognormal (+34.9)." in out
    assert "as R fitdistrplus::gofstat()" in out


def test_compare_distributions_leaves_out_what_cannot_fit(capsys):
    data = pd.DataFrame({"x": [-1.5, 0.2, 1.1, 2.3, 0.8, -0.4, 1.7], "g": list("aaabbbb")})
    t = make(data, "pandas")
    frame = t.compare_distributions("x", distributions=("normal", "gamma", "logistic"))._frame.to_pandas()
    assert set(frame["distribution"]) == {"normal", "logistic"}
    assert "gamma was left out" in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="cannot be compared"):
        t.compare_distributions("x", distributions=("normal", "poisson"))
    grouped = t.compare_distributions("x", distributions=("normal", "logistic"), for_each="g")._frame.to_pandas()
    assert list(grouped["g"]) == ["a", "a", "b", "b"]


def test_shape_test_statistic_matches_goftest_and_p_value_the_r_bootstrap(days):
    # goftest::ad.test(d, "pgamma", shape = 2.12734096971, rate = 0.00446623248479)$statistic = 1.05483345026.
    # p-value: parametric bootstrap in R, 20000 gamma samples each refitted exactly (ref_boot.R): 0.0103.
    result = shape_test(days, "gamma", column="days")  # 9999 samples, seed 1 (about 2 s)
    assert result["statistic"] == pytest.approx(1.05483345026, rel=1e-8)
    assert result["p-value"] == pytest.approx(0.0103, abs=0.02)
    assert result["distribution"].parameters()["shape"] == pytest.approx(GAMMA["shape"], rel=1e-10)
    assert 'The values of "days" do not look like a gamma distribution' in result["sentence"]
    assert "seed=1" in result["method"] and 'ad.test(x, "pgamma"' in result["method"]


def test_shape_test_other_shapes_and_errors(days):
    weibull = shape_test(days, "weibull", n_mc_samples=19)
    # goftest::ad.test(d, "pweibull", shape = 1.68720733202, scale = 531.396176566)
    assert weibull["statistic"] == pytest.approx(0.656369498, rel=1e-6)
    lognormal = shape_test(days, "lognormal", n_mc_samples=19)
    assert lognormal["statistic"] == pytest.approx(2.96302269146, rel=1e-8)  # ad.test(d, "plnorm", ...)
    again = shape_test(days, "lognormal", n_mc_samples=19)
    assert again["p-value"] == lognormal["p-value"]  # the seed repeats
    with pytest.raises(SimplydataError, match="count distribution"):
        shape_test(COUNTS, "poisson")
    with pytest.raises(SimplydataError, match="does not know"):
        shape_test(days, "gamm")


# ====================================================================== compare_shape_to with any shape
def test_compare_shape_to_any_shape_uses_the_shape_test(engine, capsys):
    trial = sdata.example("trial")
    t = make(trial, engine)
    result = row(t.compare_shape_to("days", "gamma"))
    out = capsys.readouterr().out
    assert result["statistic"] == pytest.approx(1.05483345026, rel=1e-8)  # goftest::ad.test, as above
    assert result["test"].startswith("Anderson-Darling") and result["seed"] == 1
    assert result["Monte Carlo samples"] == 9999
    assert 'The values of "days" do not look like a gamma distribution' in out
    assert "Test: Anderson-Darling test of a fitted gamma shape" in out and "seed=1" in out
    assert out.count("Test: Anderson-Darling") == 1


def test_compare_shape_to_any_shape_for_each_group(capsys):
    t = make(sdata.example("trial"), "pandas")
    frame = t.compare_shape_to("days", "lognormal", for_each="treatment")._frame.to_pandas()
    out = capsys.readouterr().out
    assert list(frame["treatment"]) == ["new medicine", "usual care"]
    for name, part in sdata.example("trial").groupby("treatment"):
        alone = shape_test(part["days"].to_numpy(dtype=float), "lognormal", column="days")
        got = frame.loc[frame["treatment"] == name].iloc[0]
        assert got["statistic"] == pytest.approx(alone["statistic"], rel=1e-12)
        assert got["p-value"] == alone["p-value"]  # the same seed, so the same p-value
        assert f"  {name}: " in out
    assert out.count("Test: Anderson-Darling") == 1


def test_compare_shape_to_normal_keeps_shapiro_wilk_and_unknown_shapes_are_explained(capsys):
    t = make(sdata.example("marks"), "pandas")
    assert row(t.compare_shape_to("marks", "normal"))["test"] == "Shapiro-Wilk test"
    with pytest.raises(SimplydataError, match="does not know"):
        t.compare_shape_to("marks", "gamm")
    with pytest.raises(SimplydataError, match="count distribution"):
        t.compare_shape_to("marks", "poisson")
    with pytest.raises(SimplydataError, match="seed="):
        t.compare_shape_to("marks", "gamma", seed="one")
