import numpy as np
import pandas as pd
import pytest

from qsarcons.consensus import calc_accuracy
from qsarcons.ensemble import CaruanaSearch


def make_models(task, seed=0):
    """Predictions of 30 models for a continuous or a binary target."""
    rng = np.random.default_rng(seed)
    if task == "continuous":    # biased, noisy models: a smaller consensus than the largest one can win
        y = rng.normal(size=60)
        x = {f"m{i}": y * rng.uniform(0.7, 1.3) + rng.uniform(-0.6, 0.6) + rng.normal(0, 0.2 + 0.1 * rng.random() * i, 60)
             for i in range(30)}
    else:                       # classifiers that get worse with i
        y = rng.integers(0, 2, 120)
        x = {f"m{i}": np.where(rng.random(120) < 0.05 + 0.01 * i, 1 - y, y) for i in range(30)}
    return pd.DataFrame(x), y


@pytest.mark.parametrize("task", ["continuous", "binary"])
def test_runs_for_both_tasks(task):
    x, y = make_models(task)
    models = CaruanaSearch(seed=0, cons_size="auto").run(x, y)
    assert len(models) > 0 and set(models) <= set(x.columns)


@pytest.mark.parametrize("task", ["continuous", "binary"])
def test_auto_mode_predicts_with_the_weights_of_the_winning_size(task):
    x, y = make_models(task)
    search = CaruanaSearch(seed=0, cons_size="auto")
    models = search.run(x, y)

    assert set(search.weights_) == set(models)
    assert sum(search.weights_.values()) == pytest.approx(1.0)

    scores = []
    for size in search.cons_size_candidates:
        single = CaruanaSearch(seed=0, cons_size=size)
        chosen = single.run(x, y)
        scores.append(calc_accuracy(y, single.predict(x[chosen])))
    assert calc_accuracy(y, search.predict(x[models])) == pytest.approx(max(scores))
