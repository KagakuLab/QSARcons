import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from qsarcons.consensus import SystematicSearch, calc_accuracy
from qsarcons.hopt import StepwiseHopt
from qsarcons.utils import detect_task


@pytest.mark.parametrize("y, expected", [
    ([0, 1, 1, 0], "binary"),
    ([0.0, 1.0, 1.0], "binary"),
    (pd.Series([1.0, 1.0]), "binary"),                  # a single class present is still the binary task
    ([0.5, 1.25, 3.0], "continuous"),
    ([24.0, 27.0, 31.0, 28.0], "continuous"),           # whole-number regression targets (sklearn: "multiclass")
    (np.array([5, 6, 7]), "continuous"),
])
def test_detect_task(y, expected):
    assert detect_task(y) == expected


def test_calc_accuracy_is_r2_for_whole_number_regression_targets():
    y = [24.0, 27.0, 31.0, 28.0]
    assert calc_accuracy(y, y) == 1.0


def test_calc_accuracy_is_f1_for_binary_targets():
    assert calc_accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(2 / 3)


def test_consensus_search_runs_on_whole_number_regression_targets():
    y = np.array([24.0, 27.0, 31.0, 24.0, 24.0, 28.0, 26.0, 30.0])
    x = pd.DataFrame({"m1": y + 0.5, "m2": y - 1.0, "m3": y + 2.0})
    assert list(SystematicSearch(cons_size=1).run(x, y)) == ["m1"]


def test_hyperparameter_search_runs_on_whole_number_regression_targets():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.round(x[:, 0] * 5 + 20)
    StepwiseHopt(Ridge(), {"alpha": [0.1, 1.0]}, verbose=False).fit(x, y)
