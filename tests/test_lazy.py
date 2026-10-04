import os

import numpy as np
import pandas as pd
import pytest
from conftest import MockEstimator

import qsarcons.modelling.lazy as lazy_mod
from qsarcons.modelling.lazy import (
    LazyML,
    baseline_prediction,
    build_model,
    calc_descriptors,
    clean_descriptors,
    scale_descriptors,
    validate_smiles,
)
from qsarcons.logging import FailedMolecule

# ---------------------------------------------------------------------------
# validate_smiles
# ---------------------------------------------------------------------------

def test_validate_smiles_keeps_valid_ones_as_strings():
    result = validate_smiles(["CCO", "c1ccccc1"])
    assert result == ["CCO", "c1ccccc1"]


def test_validate_smiles_wraps_unparseable_smiles():
    result = validate_smiles(["CCO", "not_a_valid_smiles!!!"])
    assert result[0] == "CCO"
    assert isinstance(result[1], FailedMolecule)
    assert result[1].message == "SMILES parsing failed"


def test_validate_smiles_rejects_empty_string():
    result = validate_smiles([""])
    assert isinstance(result[0], FailedMolecule)


def test_validate_smiles_verbose_reports_summary(capsys):
    validate_smiles(["CCO", "not_a_valid_smiles!!!"], verbose=True)
    assert "Validated 1 of 2 molecules" in capsys.readouterr().out


def test_validate_smiles_quiet_prints_nothing(capsys):
    validate_smiles(["CCO"], verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# baseline_prediction
# ---------------------------------------------------------------------------

def test_baseline_prediction_continuous_is_mean():
    assert baseline_prediction([1.0, 2.0, 3.0], "continuous") == pytest.approx(2.0)


def test_baseline_prediction_binary_is_most_common_class():
    assert baseline_prediction([0, 1, 1, 1, 0], "binary") == 1


# ---------------------------------------------------------------------------
# clean_descriptors / calc_descriptors
# ---------------------------------------------------------------------------

def test_clean_descriptors_imputes_nan_with_column_mean():
    x = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    cleaned = clean_descriptors(x)
    assert cleaned.shape == (3, 2)
    assert cleaned[0, 1] == pytest.approx(5.0)  # mean of [4.0, 6.0]


def test_clean_descriptors_drops_all_nan_columns():
    x = np.array([[1.0, np.nan], [3.0, np.nan]])
    cleaned = clean_descriptors(x)
    assert cleaned.shape == (2, 1)


def test_calc_descriptors_applies_calculator_then_cleans():
    calculator = lambda smi_list, ignore_errors=True: (
        np.array([[len(s)] for s in smi_list], dtype=float), np.arange(len(smi_list))
    )
    x = calc_descriptors(["CC", "CCCC"], calculator)
    assert x.tolist() == [[2.0], [4.0]]


def test_calc_descriptors_tolerates_a_molecule_the_descriptor_cannot_featurize():
    """Reproduces a real molfeat failure mode: a molecule parses fine (passes validate_smiles) but a
    specific descriptor still can't featurize it. calc_descriptors must not propagate that as a crash -
    the failed molecule's row is left for clean_descriptors' column-mean imputation instead."""

    def flaky_calculator(smi_list, ignore_errors=True):
        # index 1 is the one this descriptor "can't handle" - molfeat drops it from both outputs
        feats = np.array([[10.0], [30.0]])
        ids = np.array([0, 2])
        return feats, ids

    x = calc_descriptors(["CC", "CCC", "CCCC"], flaky_calculator)
    assert x.shape == (3, 1)
    assert x[0, 0] == pytest.approx(10.0)
    assert x[2, 0] == pytest.approx(30.0)
    assert x[1, 0] == pytest.approx(20.0)  # imputed as the mean of the two molecules that did featurize


# ---------------------------------------------------------------------------
# scale_descriptors
# ---------------------------------------------------------------------------

def test_scale_descriptors_fits_on_train_only():
    x_train = np.array([[0.0], [10.0]])
    x_test = np.array([[20.0]])
    scaled_train, scaled_test = scale_descriptors(x_train, x_test)
    assert scaled_train.tolist() == [[0.0], [1.0]]
    assert scaled_test.tolist() == [[2.0]]  # extrapolates beyond [0, 1] since scaler was fit on train only


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def _tiny_matrices():
    x_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    x_val = np.array([[1.5]])
    x_test = np.array([[3.5]])
    y_train = [1.1, 2.2, 3.3, 4.4]
    y_val = [1.5]
    return x_train, x_val, x_test, y_train, y_val


def test_build_model_hopt_false_skips_tuning():
    x_train, x_val, x_test, y_train, y_val = _tiny_matrices()
    pred_train, pred_val, pred_test = build_model(
        x_train, x_val, x_test, y_train, y_val, MockEstimator, hopt=False, task="continuous"
    )
    assert len(pred_train) == 4
    assert len(pred_val) == 1
    assert len(pred_test) == 1


def test_build_model_hopt_true_uses_reduced_grid_for_random_forest(monkeypatch):
    from sklearn.ensemble import RandomForestRegressor

    seen_grids = []
    orig_init = lazy_mod.StepwiseHopt.__init__

    def spy_init(self, estimator, param_grid, verbose=True):
        seen_grids.append(param_grid)
        orig_init(self, estimator, param_grid, verbose)

    monkeypatch.setattr(lazy_mod.StepwiseHopt, "__init__", spy_init)

    x_train, x_val, x_test, y_train, y_val = _tiny_matrices()
    build_model(x_train, x_val, x_test, y_train, y_val, RandomForestRegressor, hopt=True, task="continuous")

    assert seen_grids == [lazy_mod.REDUCED_PARAM_GRID_REGRESSORS["RandomForestRegressor"]]


def test_build_model_hopt_true_uses_default_grid_for_ridge(monkeypatch):
    from sklearn.linear_model import Ridge
    from qsarcons.hopt import DEFAULT_PARAM_GRID_REGRESSORS

    seen_grids = []
    orig_init = lazy_mod.StepwiseHopt.__init__

    def spy_init(self, estimator, param_grid, verbose=True):
        seen_grids.append(param_grid)
        orig_init(self, estimator, param_grid, verbose)

    monkeypatch.setattr(lazy_mod.StepwiseHopt, "__init__", spy_init)

    x_train, x_val, x_test, y_train, y_val = _tiny_matrices()
    build_model(x_train, x_val, x_test, y_train, y_val, Ridge, hopt=True, task="continuous")

    assert seen_grids == [DEFAULT_PARAM_GRID_REGRESSORS["Ridge"]]


# ---------------------------------------------------------------------------
# LazyML.__init__
# ---------------------------------------------------------------------------

def test_lazyml_requires_output_folder():
    with pytest.raises(ValueError):
        LazyML(task="continuous", output_folder=None)


def test_lazyml_clears_stale_output_folder(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "stale.txt").write_text("stale")

    LazyML(task="continuous", output_folder=str(target))
    assert os.path.isdir(target)
    assert not os.path.exists(target / "stale.txt")


def test_lazyml_selects_estimator_set_by_task(tmp_path):
    lazy_c = LazyML(task="continuous", output_folder=str(tmp_path / "c"))
    lazy_b = LazyML(task="binary", output_folder=str(tmp_path / "b"))
    assert lazy_c.ESTIMATORS is lazy_mod.REGRESSORS
    assert lazy_b.ESTIMATORS is lazy_mod.CLASSIFIERS


# ---------------------------------------------------------------------------
# LazyML.run - full flow with fast monkeypatched descriptors/estimators
# ---------------------------------------------------------------------------

def _fast_descriptors():
    def calc(smi_list, ignore_errors=True):
        feats = np.array([[len(s)] for s in smi_list], dtype=float)
        return feats, np.arange(len(smi_list))
    return {"FakeDesc": calc}


def test_lazyml_run_continuous_verbose(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "not_a_valid_smiles!!!", "CCN"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCC"]
    y_val = [5.5]
    smi_test = ["CCCl"]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    result_train, result_val, result_test = lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    assert len(result_train) == 4  # rows are never dropped, even on SMILES-parse failure
    assert len(result_val) == 1
    assert len(result_test) == 1
    assert "FakeDesc|Mock" in result_test.columns
    assert list(result_test.columns) == ["SMILES", "FakeDesc|Mock"]
    assert list(result_val.columns) == ["SMILES", "Y_TRUE", "FakeDesc|Mock"]

    captured = capsys.readouterr()
    assert "Step-1. SMILES validation" in captured.out
    assert "Step-2. Descriptor calculation" in captured.out
    assert "Step-3. Individual model training" in captured.out
    assert "Validated 5 of 6 molecules" in captured.out
    assert "FakeDesc: done" in captured.out
    assert "[1/1] FakeDesc|Mock" in captured.out
    assert (tmp_path / "out" / "train.csv").exists()
    assert (tmp_path / "out" / "val.csv").exists()
    assert (tmp_path / "out" / "test.csv").exists()


def test_lazyml_run_binary_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [0, 1, 0, 1]
    smi_val = ["CCCl"]
    y_val = [0]

    lazy = LazyML(task="binary", hopt=False, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])


def test_lazyml_run_splits_are_taken_as_given(monkeypatch, tmp_path):
    """LazyML performs no train/val split of its own - it trusts the split it's handed."""
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl", "CCF"]
    y_val = [5.5, 6.6]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    result_train, result_val, _ = lazy.run(smi_train, y_train, smi_val, y_val, [])

    assert list(result_train["SMILES"]) == smi_train
    assert list(result_val["SMILES"]) == smi_val


def test_lazyml_run_test_predictions_use_baseline_for_failed_molecules(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 3.3, 2.2, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]
    smi_test = ["CCF", "not_a_valid_smiles!!!"]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    _, _, result_test = lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    assert len(result_test) == 2  # test rows are never dropped, even on failure
    baseline = baseline_prediction(y_train, "continuous")
    assert result_test["FakeDesc|Mock"].iloc[1] == pytest.approx(baseline)

    captured = capsys.readouterr()
    assert "1 test molecule(s) could not be processed" in captured.out


def test_lazyml_run_train_predictions_use_baseline_for_failed_molecules(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "not_a_valid_smiles!!!", "CCN"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    result_train, _, _ = lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])

    valid_y_train = [1.1, 2.2, 4.4]  # the failed molecule's own target is excluded from the baseline
    baseline = baseline_prediction(valid_y_train, "continuous")

    assert len(result_train) == 4  # train rows are never dropped, even on failure
    assert result_train["FakeDesc|Mock"].iloc[2] == pytest.approx(baseline)

    captured = capsys.readouterr()
    assert "1 training molecule(s) could not be processed" in captured.out


def test_lazyml_run_silent_when_nothing_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])

    assert "could not be processed" not in capsys.readouterr().out


def test_lazyml_run_descriptors_calculated_once_across_splits(monkeypatch, tmp_path):
    """Descriptors are computed for train+val+test together in one call per descriptor type, not once
    per split - this is what guarantees consistent columns across splits after NaN-column cleanup,
    and is also what lets calc_descriptors' ignore_errors path share one imputation basis."""
    calls = []

    real_calc_descriptors = lazy_mod.calc_descriptors

    def _counting_calc_descriptors(smi_list, calculator):
        calls.append(len(smi_list))
        return real_calc_descriptors(smi_list, calculator)

    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "calc_descriptors", _counting_calc_descriptors)

    smi_train = ["CCO", "c1ccccc1", "CCN"]
    y_train = [1.1, 2.2, 3.3]
    smi_val = ["CCC"]
    y_val = [4.4]
    smi_test = ["CCCl", "CCF"]

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    # one call, covering every molecule across all three splits
    assert calls == [len(smi_train) + len(smi_val) + len(smi_test)]


def test_lazyml_run_empty_test_set(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    lazy = LazyML(task="continuous", hopt=False, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator}
    _, _, result_test = lazy.run(
        ["CCO", "c1ccccc1", "CCN", "CCC"], [1.1, 2.2, 3.3, 4.4], ["CCCl"], [5.5], []
    )
    assert len(result_test) == 0
    assert list(result_test.columns) == ["SMILES", "FakeDesc|Mock"]


# ---------------------------------------------------------------------------
# DESCRIPTORS / REGRESSORS / CLASSIFIERS - built-in configuration dicts
# ---------------------------------------------------------------------------

def test_default_descriptors_are_callable():
    for name, calculator in lazy_mod.DESCRIPTORS.items():
        assert callable(calculator), name


def test_reduced_grids_only_cover_mlp_and_random_forest():
    assert set(lazy_mod.REDUCED_PARAM_GRID_REGRESSORS) == {"MLPRegressor", "RandomForestRegressor"}
    assert set(lazy_mod.REDUCED_PARAM_GRID_CLASSIFIERS) == {"MLPClassifier", "RandomForestClassifier"}
