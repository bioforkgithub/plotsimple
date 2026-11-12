import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from plotsimple import _errors


@pytest.fixture(autouse=True)
def fresh_old_name_notes():
    """Each test starts like a new session: an old name prints its note again the first time."""
    _errors._NOTED.clear()
    yield


@pytest.fixture
def marks():
    """A small class-marks table used by many tests."""
    rng = np.random.default_rng(1)
    names = ["Asha", "Ben", "Chen", "Dina"]
    rows = []
    for test in (1, 2, 3):
        for cls in ("A", "B"):
            for name in names:
                rows.append({"name": name, "class": cls, "test": test,
                             "marks": float(rng.integers(30, 100)), "height": float(rng.normal(150, 8))})
    return pd.DataFrame(rows)
