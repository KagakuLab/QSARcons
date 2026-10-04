import os
import re

import pandas as pd
import pytest
from conftest import MockEstimator

import qsarcons.modelling.lazy as lazy_mod
import qsarcons.modelling.meta as meta_mod
from qsarcons.modelling.meta import ConsensusClassifier, ConsensusEstimator, ConsensusRegressor


class FakeConsensusSearch:
    """Stand-in for a real ConsensusSearch (GeneticSearch/RandomSearch/SystematicSearch) - real search
    is unnecessary overhead for testing meta.py's own orchestration, which is all these tests target."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, x_val, true_val):
        return list(x_val.columns)

    def predict(self, x_subset):
        return list(x_subset.mean(axis=1))


class FakeConsensusSearchBadConsensus:
    """Returns a consensus referencing a model column that doesn't exist."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, x_val, true_val):
        return ["Missing|Model"]

    def predict(self, x_subset):
        return list(x_subset.mean(axis=1))


def _fast_descriptors():
    import numpy as np
    def calc(smi_list, ignore_errors=True):
        return np.array([[len(s)] for s in smi_list], dtype=float), np.arange(len(smi_list))
    return {"FakeDesc": calc}


def _patch_fast_pipeline(monkeypatch, method="genetic", classifier=False, search_class=FakeConsensusSearch):
    monkeypatch.setitem(meta_mod.CONSENSUS_BUILDERS, method, lambda cons_size, verbose: search_class())
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    if classifier:
        monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator})
    else:
        monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_regressor_and_classifier_share_base_class():
    assert issubclass(ConsensusRegressor, ConsensusEstimator)
    assert issubclass(ConsensusClassifier, ConsensusEstimator)


def test_regressor_forces_continuous_task(tmp_path):
    """ConsensusRegressor's whole reason to exist: its LazyML is built with task="continuous" directly,
    rather than inferring it from the target values."""
    model = ConsensusRegressor(output_folder=str(tmp_path / "out"))
    assert model._lazy_model.task == "continuous"
    assert model._lazy_model.ESTIMATORS is lazy_mod.REGRESSORS


def test_classifier_forces_binary_task(tmp_path):
    model = ConsensusClassifier(output_folder=str(tmp_path / "out"))
    assert model._lazy_model.task == "binary"
    assert model._lazy_model.ESTIMATORS is lazy_mod.CLASSIFIERS


def test_init_default_output_folder_is_timestamped():
    model = ConsensusRegressor()
    assert re.fullmatch(r"qsarcons_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}", model.output_folder)


def test_init_rejects_unknown_consensus_method():
    with pytest.raises(ValueError, match="Unknown consensus method"):
        ConsensusRegressor(consensus="not_a_real_method")


@pytest.mark.parametrize("method", ["genetic", "random", "systematic"])
def test_init_accepts_every_known_consensus_method(method, tmp_path):
    model = ConsensusRegressor(consensus=method, output_folder=str(tmp_path / method))
    assert model.consensus == method


# ---------------------------------------------------------------------------
# train_predict - end to end, with a faked consensus search for speed
# ---------------------------------------------------------------------------

def test_train_predict_end_to_end_regression(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    smiles_test = ["CCF"]

    model = ConsensusRegressor(
        consensus="genetic", hopt=False, output_folder=str(tmp_path / "out"), verbose=True, random_seed=42
    )
    preds = model.train_predict(smiles_train, y_train, smiles_test)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert len(model.best_consensus) > 0

    captured = capsys.readouterr()
    assert "Step-4. Consensus search" in captured.out
    assert "Best consensus" in captured.out

    assert (tmp_path / "out" / "train.csv").exists()
    assert (tmp_path / "out" / "val.csv").exists()

    test_df = pd.read_csv(tmp_path / "out" / "test.csv")
    assert "FakeDesc|Mock" in test_df.columns
    assert len(test_df) == 1


def test_train_predict_end_to_end_classification_quiet(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch, classifier=True)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [0, 1, 0, 1, 0]
    smiles_test = ["CCF"]

    model = ConsensusClassifier(
        consensus="genetic", hopt=False, output_folder=str(tmp_path / "out"), verbose=False, random_seed=99
    )
    preds = model.train_predict(smiles_train, y_train, smiles_test)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert capsys.readouterr().out == ""


def test_random_seed_produces_different_train_val_splits(monkeypatch, tmp_path):
    """self.random_seed reaches train_test_split's random_state."""
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    smiles_test = ["CCF"]

    model_a = ConsensusRegressor(hopt=False, output_folder=str(tmp_path / "out_a"), verbose=False, random_seed=1)
    model_b = ConsensusRegressor(hopt=False, output_folder=str(tmp_path / "out_b"), verbose=False, random_seed=2)
    model_a.train_predict(smiles_train, y_train, smiles_test)
    model_b.train_predict(smiles_train, y_train, smiles_test)

    train_a = pd.read_csv(tmp_path / "out_a" / "train.csv")
    train_b = pd.read_csv(tmp_path / "out_b" / "train.csv")
    assert list(train_a["SMILES"]) != list(train_b["SMILES"])


def test_train_predict_uses_the_requested_consensus_method(monkeypatch, tmp_path):
    """Swapping consensus="systematic" must route through CONSENSUS_BUILDERS["systematic"], not genetic."""
    called_with = []

    class TrackedFakeSearch(FakeConsensusSearch):
        def __init__(self, *args, **kwargs):
            called_with.append("systematic")

    _patch_fast_pipeline(monkeypatch, method="systematic", search_class=TrackedFakeSearch)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]

    model = ConsensusRegressor(consensus="systematic", hopt=False, output_folder=str(tmp_path / "out"), verbose=False)
    model.train_predict(smiles_train, y_train, ["CCF"])

    assert called_with == ["systematic"]


def test_train_predict_raises_when_consensus_columns_missing(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch, search_class=FakeConsensusSearchBadConsensus)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    model = ConsensusRegressor(output_folder=str(tmp_path / "out"), hopt=False, verbose=False)

    with pytest.raises(KeyError):
        model.train_predict(smiles_train, y_train, ["CCF"])
